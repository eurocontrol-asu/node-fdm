import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchdiffeq import odeint

from node_fdm.data.loader import get_train_val_data
from node_fdm.architectures.mapping import get_architecture_from_name
from node_fdm.models.flight_dynamics_model import FlightDynamicsModel
from node_fdm.models.batch_neural_ode import BatchNeuralODE
from utils.learning.loss import get_loss


class ODETrainer:
    def __init__(
        self,
        data_df,
        model_config,
        model_dir,
        num_workers=4,
        load_parallel=True,
        train_val_num=(5000, 500)
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        architecture, self.model_cols, custom_fn = get_architecture_from_name(model_config["architecture_name"])
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = self.model_cols
        self.model_dir = model_dir / model_config["model_name"]
        os.makedirs(self.model_dir, exist_ok=True)
        self.architecture = architecture
        self.model_config = model_config
        self.architecture_name = model_config["architecture_name"]
        self.model_params = model_config["model_params"]

        # ---- Load dataset only once! ----
        self.train_dataset, self.val_dataset = get_train_val_data(
            data_df,
            self.model_cols,
            shift=model_config['shift'],
            seq_len=model_config['seq_len'],
            custom_fn=custom_fn,
            load_parallel=load_parallel,
            train_val_num = train_val_num
        )
        self.step = model_config['step']
        self.num_workers = num_workers

        # Means and stds from dataset (for model normalization)
        self.stats_dict = self.train_dataset.stats_dict

        # ---- Model & optimizer ----
        self.model = self.get_or_create_model(*model_config["loading_args"])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay']
        )
        self.epoch = 1
        self.save_meta()


    def get_or_create_model(self, load=False, load_loss=False):
        self.best_val_loss = float("inf")
        if load and os.path.exists(self.model_dir /  "meta.json"):
            model = self.load_best_checkpoint(load_loss=load_loss)
        else:
            print("Creating new model.")
            model = FlightDynamicsModel(
                self.architecture,
                self.stats_dict,
                self.model_cols,
                model_params=self.model_params,
            ).to(self.device)
        return model

    def load_best_checkpoint(self, load_loss=False):
        # Create model with loaded architecture and params
        model = FlightDynamicsModel(
            self.architecture,
            self.stats_dict,
            self.model_cols,
            model_params=self.model_params,
        ).to(self.device)

        # Load each layer state dict if checkpoint exists
        for name in model.layers_name:
            checkpoint = self.load_layer_checkpoint(name)
            if checkpoint is not None:
                model.layers_dict[name].load_state_dict(
                    checkpoint["layer_state"], strict=False
                )
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                self.epoch = checkpoint.get("epoch", 0)
            else:
                # Initialize defaults if no checkpoint found
                best_val_loss = float("inf")
                self.epoch = 0

        # Optionally aggregate best_val_loss over all layers (min or average)
        if load_loss:
            self.best_val_loss = best_val_loss

        print("Best val loss per layer:", self.best_val_loss)
        print(f"Loaded modular model from {self.model_dir}")

        return model

    def load_layer_checkpoint(self, layer_name):
        path = os.path.join(self.model_dir, f"{layer_name}.pt")
        if not os.path.exists(path):
            print(f"No checkpoint found for layer {layer_name}, skipping load.")
            return None
        else:
            print(f"checkpoint found for layer {layer_name}")
        checkpoint = torch.load(path, map_location=self.device)
        return checkpoint

    def save_meta(self):
        saved_stats_dict = {str(col) : value for col, value in self.stats_dict.items()}

        meta_dict = {
            "architecture_name": self.architecture_name,
            "model_params": self.model_config["model_params"],
            "step": self.model_config["step"],
            "shift": self.model_config["shift"],
            "lr": self.model_config["lr"],
            "seq_len": self.model_config["seq_len"],
            "batch_size": self.model_config["batch_size"],
            "stats_dict": saved_stats_dict,
        }

        with open(self.model_dir / "meta.json", "w") as f:
            json.dump(meta_dict, f, indent=4)

    def save_layer_checkpoint(self, layer_name, epoch):
        layer = self.model.layers_dict[layer_name]
        save_dict = {
            "layer_state": layer.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),  # or per-layer optimizer state if you have
            "best_val_loss": self.best_val_loss,
            "epoch": self.epoch + epoch,
        }
        torch.save(save_dict, self.model_dir / f"{layer_name}.pt")

    def save_model(self, epoch):
        for name in self.model.layers_name:
            self.save_layer_checkpoint(name, epoch)

    def norm_vect(self, vect, col):
        return (vect - self.stats_dict[col]["mean"]) / (
            self.stats_dict[col]["std"] + 1e-6
        )

    def cat_to_dict_vects(self, vect_list, col_list, alpha_dict, normalize=True):

        def modifier(el, col):
            if (col.normalize_mode == "normal") & (normalize):
                return self.norm_vect(el, col)
            else:
                return el

        vects = torch.cat(vect_list, dim=2)

        vects_dict = {
            col: (alpha_dict[col] if col in alpha_dict.keys() else 1.0)
            * modifier(vects[..., i], col).unsqueeze(-1)
            for i, col in enumerate(col_list)
        }
        return vects_dict

    def ode_step(
        self,
        x_seq,
        u_seq,
        e_seq,
        method,
        alpha_dict,
    ):
        seq_len = x_seq.shape[1]

        assert not torch.isnan(x_seq).any(), "NaN in x_seq"
        assert not torch.isnan(u_seq).any(), "NaN in u_seq"
        assert not torch.isnan(e_seq).any(), "NaN in e_seq"

        x0 = x_seq[:, 0, :]

        t_grid = torch.arange(
            0, seq_len * self.step, self.step, dtype=torch.float32, device=self.device
        )

        func = BatchNeuralODE(self.model, u_seq, e_seq, t_grid)

        odeint(func, x0, t_grid, method=method)

        vects = torch.cat([x_seq, u_seq, e_seq], dim=2)
        vect_dict = {
            col: vects[..., i].unsqueeze(-1)
            for i, col in enumerate(
                self.x_cols + self.u_cols + self.e0_cols + self.e_cols
            )
        }

        vects_dict = dict()

        monitor_cols = self.x_cols + self.e_cols

        for case in ["true", "pred"]:
            if case == "pred":
                vect_list = [
                    self.model.history[col].unsqueeze(-1) for col in monitor_cols
                ]
            else:
                vect_list = [vect_dict[col][:, 1:] for col in monitor_cols]

            vects_dict[case] = self.cat_to_dict_vects(
                vect_list,
                monitor_cols,
                alpha_dict=alpha_dict,
            )

        true_vect = torch.cat([vects_dict["true"][col] for col in monitor_cols], dim=2)
        pred_vect = torch.cat([vects_dict["pred"][col] for col in monitor_cols], dim=2)
        return true_vect, pred_vect, monitor_cols

    def compute_loss_ode_step(
        self,
        batch,
        alpha_dict,
        method="rk4"
    ):
        x_seq, u_seq, e_seq, _ = [b.to(self.device) for b in batch]
        true_vect, pred_vect, final_cols = self.ode_step(
            x_seq,
            u_seq,
            e_seq,
            method,
            alpha_dict,
        )

        loss = 0.0
        for i, col in enumerate(alpha_dict.keys()):
            loss_fn = get_loss(col.loss_name)

            assert not torch.isnan(pred_vect[..., i]).any(), "NaN in pred_vect"
            assert not torch.isnan(true_vect[..., i]).any(), "NaN in true_vect"

            loss += loss_fn(pred_vect[..., i], true_vect[..., i])

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf in loss!")

        return loss

    def train(
        self,
        epochs=800,
        batch_size=512,
        val_batch_size=10000,
        scheduler=None,
        method="rk4",
        alpha_dict = None,
    ):
        # --- DataLoaders ---
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if alpha_dict is None:
            alpha_dict = {
                col: 1.0 for col in self.x_cols
            }

        self.stats_dict = self.train_dataset.stats_dict

        # === NEW: loss tracking ===
        losses = []
        loss_csv_path = os.path.join(self.model_dir, "training_losses.csv")
        fig_path = os.path.join(self.model_dir, "training_curve.png")

        for epoch in range(epochs):
            # --- TRAIN LOOP ---
            self.model.train()
            total_loss, total_batches = 0, 0
            for batch in self.train_loader:
                loss = self.compute_loss_ode_step(
                    batch, alpha_dict=alpha_dict, method=method
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                total_batches += 1
            avg_train_loss = total_loss / total_batches

            # --- VALIDATION LOOP ---
            self.model.eval()
            val_loss, val_batches = 0, 0
            with torch.no_grad():
                for batch in self.val_loader:
                    loss = self.compute_loss_ode_step(
                        batch, alpha_dict=alpha_dict, method=method
                    )
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches

            if scheduler is not None:
                scheduler.step(avg_val_loss)

            # Log losses
            losses.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

            print(
                f"Epoch {epoch+1}/{epochs} | train loss: {avg_train_loss:.5f} | val loss: {avg_val_loss:.5f}"
            )

            # --- SAVE BEST MODEL ---
            if avg_val_loss < self.best_val_loss:
                print(f"  New best validation loss: {avg_val_loss:.5f}. Saving model.")
                self.best_val_loss = avg_val_loss
                self.save_model(epoch)

        # === SAVE LOSSES WITH PANDAS ===
        df_losses = pd.DataFrame(losses)
        df_losses.to_csv(loss_csv_path, index=False)
        print(f"✅ Saved training log to {loss_csv_path}")

        # === PLOT TRAINING CURVE ===
        plt.figure(figsize=(7, 4))
        plt.semilogy(
            df_losses["epoch"],
            df_losses["train_loss"],
            label="Training loss",
            color="#1f77b4",
            linewidth=2,
        )
        plt.semilogy(
            df_losses["epoch"],
            df_losses["val_loss"],
            label="Validation loss",
            color="#ff7f0e",
            linewidth=2,
            linestyle="--",
        )

        plt.title("Training and validation losses", fontsize=13)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss (log scale)", fontsize=11)
        plt.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.7)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"✅ Saved training curve to {fig_path}")
