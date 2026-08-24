# meta_optimization_experiments
# Emergent Layer-Wise Schedule & Noise Optimization in Short-Horizon Meta-Learning

An empirical study exploring joint meta-optimization of layer-specific learning rates ($\eta$) and injection noise ($\sigma$) across unrolled training steps.

## Overview

This repository investigates how a meta-optimizer learns to dynamically schedule hyper-parameters across distinct neural network layers (Convolutional vs. Fully Connected) when unrolled over short ($K=8$) and extended ($K=24$) optimization steps. 

By parameterizing layer-wise learning rates and relative noise scales, we observe emergent optimization behaviors, such as late-stage learning rate surges in short-horizon regimes and layer-differentiated decay during extended training horizons.

## Key Findings

* **Short Horizon Dynamics ($K=8$):** When constrained to ultra-short horizons, the meta-optimizer learns a late-stage surge in learning rate to maximize loss descent within the strict step limit.
* **Extended Horizon Dynamics ($K=24$):** Over 24 unrolled steps, the model converges on a 4-phase strategy, achieving **>95% test accuracy**:
  1. *Aggressive Initial Sprint:* High learning rates across all layers ($\eta \approx 0.055$) to escape initialization.
  2. *Feature Exploration:* High noise injection in early layers (`conv1`) to prevent premature convergence on local pixel artifacts.
  3. *Late-Stage Cooling:* Sharp decay of learning rates in late layers (`conv2`, `fc`) to stabilize decision boundaries.
  4. *Terminal Settling:* Complete noise shutdown in classification layers while maintaining minor feature flexibility in early convolutional layers.
* **Hessian & Curvature Limits:** Early-step learning rates plateau near $\eta \approx 0.055$, corresponding to the local curvature stability bound ($\eta < \frac{2}{\lambda_{\max}}$) of the loss surface.

## Repository Contents

* `meta_optimization_experiments.ipynb`: Interactive Google Colab notebook containing code blocks for parameter scheduling, inner-loop unrolling, gradient clipping guardrails, and visualization heatmaps.

## Acknowledgments & Credits

* **Experiment Design & Implementation:** Tanner Eckmann
* **Code Collaborator:** Google Gemini (assisted with code block implementations, numerical stabilization routines, and analytical framing)
