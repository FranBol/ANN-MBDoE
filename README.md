# **Gradient-enhanced neural networks for model parameter estimation applied to flow chemistry automated platforms**

![Workflow](Images/GeNN-MBDoE.png)

## Overview
 
This repository contains the code and data accompanying the paper:
 
> **Gradient-enhanced neural networks for model parameter estimation applied to flow chemistry automated platforms**
 
The core idea is to train **gradient-enhanced neural networks**. These artificial neural networks learn not only the system outputs but also their gradients with respect to inputs. The resulting surrogate models can then be embedded directly into a **Model-Based Design of Experiments (MBDoE)** framework to drive efficient parameter estimation and experimental campaign design. The primary motivation behind this work is to address scenarios where the underlying mechanistic model is computationally expensive to evaluate. By replacing it with a fast and differentiable surrogate, the framework can rapidly propose new experimental conditions that maximize information gain and improve kinetic parameter estimation.

### 1. Process models (`Data/`)
 
Two systems are considered:
 
- **Illustrative example**: the governing equations are defined analytically in the paper. The corresponding dataset is `data_example_dispersion.csv`.
- **Case study**: a complex first-principles model (`Discrete_inj_flow_model.py`) describes the system. `Adjoint_case_study.py` provides a way to compute output sensitivities with respect to inputs via the adjoint method. The corresponding dataset is `data_case_study.csv`.

### 2. Surrogate model training (`NN_models/`)
 
For each system, both a **standard ANN** and a **gradient-enhanced ANN** are trained as computationally cheap surrogates. Training is handled by:
- `Training_example.py` — for the illustrative example
- `Training_case_study.py` — for the case study

The saved models can subsequently be used as surrogate models within a MBDoE optimisation loop.

### 3. Sequential experimental design (`Sequential planning/`)
 
`MBDoE.py` embeds the trained surrogates into a the MBDoE framework. The Fisher Information Matrix is computed using the ANN gradients and the next D-optimal experimental conditions are suggested to improve kinetic parameter estimation.
