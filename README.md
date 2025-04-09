# Semantic Robustness Verification for Neural Networks using Image Similarity

**Author:** *Jannick Strobel*

## Overview

This tool verifies and repairs robustness properties of PyTorch neural networks using image similarity metrics.

It currently supports the **Structural Similarity Index (SSIM)** to define perturbation sets around a reference image. Verification ensures that all sufficiently similar images (under SSIM) are classified consistently by the model.

> SSIM captures **luminance**, **contrast**, and **structural** similarity between two images. For details, see [SSIM Definition](#ssim-definition).

---

## Installation

You need a working [conda](https://docs.conda.io/en/latest/) installation.

Run the install script from the root directory:

```bash
bash install/install_scip-env-conda.sh
conda activate scip-env-conda
```

This sets up the environment `scip-env-conda` using `install/scip-env-conda.yaml`. \
⚠️ The script will **delete** any existing conda environment with that name.

---

## Running an Experiment

To run a verification experiment:

```bash
python main.py data/example_params/params_mnist.json
```

The system will either train a model or load an existing one, then perform robustness verification and repair based on SSIM similarity to a reference image.

---

## Parameter File Format

The experiment is configured via a JSON file. Below is an example and explanations.

### Example: MNIST, class "2", 4×4 input, 6-6-6 hidden layers

```json
{
  "path": "./data/experiments/mnist",
  "dimension": [1, 4, 4],

  "ri": {
    "path": "./data/ri/mnist/2.png",
    "class_idx": 2
  },

  "model": {
    "dataset": "MNIST",
    "hidden_layer": [6, 6, 6],
    "load_model": true,
    "path": "./data/models/mnist",
    "data_path": "./__data__",
    "selected_classes": [1, 3, 5, 7, 9],
    "epochs": 50,
    "batch_size": 32
  },

  "minlp": {
    "distance": "ssim",
    "threshold": 0.99,
    "timeout": 86400,
    "tolerance": 4
  },

  "repair": {
    "batch_size": 32,
    "penalty_increase": 1.1,
    "lr": 0.01
  }
}
```

---

### Model Filename Convention

If `model/path` is a directory, a model will be loaded using the following filename format:

```bash
model_i<dims>_l<layers>_o<classes>.pt
```

- `i1x4x4`: input shape (e.g. 1×4×4)
- `l6-6-6`: hidden layers with 6 neurons each
- `o5`: number of output classes

---

## SSIM Definition

The SSIM between two images \( x \) and \( y \) is defined as:

\[
\text{SSIM}(x, y) = l(x, y) \cdot c(x, y) \cdot s(x, y)
\]

Where:

- **Luminance**:  
  \( l(x, y) = \frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \)

- **Contrast**:  
  \( c(x, y) = \frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2} \)

- **Structure**:  
  \( s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3} \)

with:

- \( \mu \): mean, \( \sigma \): std dev, \( \sigma_{xy} \): covariance
- \( C_1, C_2, C_3 \): stabilising constants
