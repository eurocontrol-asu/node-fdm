<p align="center">
  <img src="images/logo.jpg" alt="Neural Ordinary Differential Equation Flight Dynamics Model" width="45%">
</p>

<p align="center">
  <em>A physics-guided Neural Ordinary Differential Equation (Neural ODE) framework for aircraft flight dynamics</em>
</p>


## ✈️ What is node-fdm?

**node-fdm** is a framework for learning and simulating **aircraft flight dynamics** using  **Neural Ordinary Differential Equations** guided by **physical constraints**.

It is designed to:

- reconstruct **aircraft trajectories** from ADS-B or QAR data  
- simulate aircraft behaviour using physics-aware latent dynamics  
- support multiple architectures (OpenSky 2025, QAR, custom)
- benchmark results against physics-based baselines such as **BADA**  

This documentation will help you install the library, understand the concepts, run the pipelines, and extend the framework with your own architectures.


# 🚀 Quick Navigation

Use this documentation as your main entry point for the project.

### 👉 If you are new
Start here:  
**Guide → Quickstart**

### 👉 If you want to run the full OpenSky pipeline  
See:  
**Guide → OpenSky 2025 Pipeline**

### 👉 If you want to train or test models  
Go to:  
**How-to → Training**  
**How-to → Inference**

### 👉 If you want to understand the architecture  
Check:  
**Concepts → Model Structure**  
**Concepts → Column Groups**

### 👉 If you need the API  
Go to:  
**Reference API**

# 📌 Legal Notice

This project is distributed under the **EUPL-1.2** licence with specific EUROCONTROL amendments (see `AMENDMENT_TO_EUPL_license.md`).  It is intended **for research purposes only** and must not be used as a regulatory tool.
