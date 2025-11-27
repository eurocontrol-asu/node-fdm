# 🏋️ ODETrainer API

The `ODETrainer` class acts as the **high-level orchestrator** for the training pipeline. It wraps PyTorch Lightning to provide a standardized interface for training Neural ODEs on flight data.

It functions as the central bridge in the pipeline: it validates the configuration, retrieves the specific model architecture from the registry, initializes the training environment, and manages the lifecycle of model checkpoints and metadata artifacts.

---

## 📘 Class Reference

::: node_fdm.ode_trainer
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true