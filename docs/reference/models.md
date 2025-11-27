# 🧠 Model Wrappers API

The `node_fdm.models` namespace provides the high-level PyTorch `nn.Module` wrappers that encapsulate the Neural ODE logic.

These classes serve as the core mathematical engine of the framework. They handle the **forward pass integration**, the **batch processing** of trajectories, and the orchestration of the various layers defined in your architecture.

---

## 📘 Class Reference

### Flight Dynamics Model (Base)
The primary wrapper used during training. It manages the input encoding, connects to the ODE solver, and handles state reconstruction.

::: node_fdm.models.flight_dynamics_model
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Batch Neural ODE
The core utility responsible for solving the system of differential equations over batches of time sequences. It interfaces with the numerical solvers (Euler, RK4).

::: node_fdm.models.batch_neural_ode
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Production Model
An optimized wrapper designed strictly for inference environments. It streamlines the forward pass by removing training-specific hooks and overhead.

::: node_fdm.models.flight_dynamics_model_prod
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true