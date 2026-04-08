# **Gradient-enhanced neural networks for model parameter estimation applied to flow chemistry automated platforms**

![Workflow](Images/GeNN-MBDoE.png)

## Overview
 
This repository contains the code and data accompanying the paper:
 
> **Gradient-enhanced neural networks for model parameter estimation applied to flow chemistry automated platforms**
 
The core idea is to train **gradient-enhanced neural networks**. These artificial neural networks learn not only the system outputs but also their gradients with respect to inputs. The resulting surrogate models can then be embedded directly into a **Model-Based Design of Experiments (MBDoE)** framework to drive efficient parameter estimation and experimental campaign design. The primary motivation behind this work is to address scenarios where the underlying mechanistic model is computationally expensive to evaluate. By replacing it with a fast and differentiable surrogate, the framework can rapidly propose new experimental conditions that maximize information gain and improve kinetic parameter estimation.

### `NN_models/`
 
Contains Python scripts to train the surrogate models presented in the paper. Each script builds and trains a specific neural network architecture (standard ANN or gradient-enhanced ANN) on the data in `Data/`. The saved models can subsequently be used as surrogate models within a MBDoE optimisation loop.
