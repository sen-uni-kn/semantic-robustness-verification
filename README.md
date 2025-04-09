# Semantic Robustness Verification for Neural Networks using Image Similarity Metrics

**Author:** *Jannick Strobel*

## Info

This tool verifies and repairs robustness specifications for neural networks in PyTorch format. \
Robustness is verified within the set of inputs that are structurally similar to a given reference image.

The only image similarity metric supported at this point is the Structural Similarity Index (SSIM).

The SSIM between two images $x$ and $y$ is defined as:

$$
\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma
$$

Typically, $\alpha = \beta = \gamma = 1$, so it simplifies to:

$$
\text{SSIM}(x, y) = l(x, y) \cdot c(x, y) \cdot s(x, y)
$$

 **Luminance Comparison**: $l(x, y) = \frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$

- $\mu_x, \mu_y$: mean intensities of $x$ and $y$  
- $C_1$: constant to stabilise the division

**Contrast Comparison**: $c(x, y) = \frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$

- $\sigma_x, \sigma_y$: standard deviations of $x$ and $y$  
- $C_2$: constant to stabilise the division

**Structure Comparison**: $s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}$

- $\sigma_{xy}$: covariance between $x$ and $y$  
- $C_3$: often set to $\frac{C_2}{2}$

## Installation

With an existing conda installation, the environment for the repair and verification experiments can be installed using the install script in the `install/` folder. \
The scripts removes an existing conda environment named `scip-env-conda`!

### Environment

The optimisation tool `scip` and its python interface `pyscipopt` can be installed through conda. \
The conda environment configuration file can be found at `install/scip-env-conda.yaml` and can be installed via `conda env create -f install/scip-env-conda.yaml`

There exists a installation script that removes and existing conda env with the same name if existent and installs the new environment.
For the install script `bash` has to be used. The script has to be executed from the parent folder of `install/`.

`bash install/install_scip-env-conda.sh` \
`conda activate scip-env-conda`

## Run Experiment

`python main.py data/example/params.json`

### Param File

The tool either trains a network according to the specification given in the params file or loads a already trained network in torch format. \
The path given at model/path can be either a ".pt" file or the path to the model directory if it is named according to the following specification:

Model filenames are structured as follows:

`model_i<dims>_l<layers>_o<classes>.pt`

- **`i<dims>`**:  
  Represents the input dimensions, joined by "x".  
  Example: `i1x4x4` means the model takes an input of shape 1×4×4 (e.g. grayscale 4×4 image).

- **`l<layers>`**:  
  Hidden layer sizes, listed in order and separated by hyphens.  
  Example: `l6-6-6` indicates three hidden layers with 6 neurons each.

- **`o<classes>`**:  
  Number of output classes.  
  Example: `o5` means the model predicts 5 classes.  

### Example Param File

This parameter file runs an experiment on an MNIST classifier with a neural network with 6x6x6 ReLU nodes and an 4x4 greyscale input.
The experiment runs repair for the given reference image with class "2".

```text
{
    "path": "./data/experiments/mnist",         # Output directory for storing experiment results (logs, metrics, etc.)

    "dimension": [1, 4, 4],                     # Input dimensions (C × H × W), e.g. 1 channel, 4x4 image

    "ri": {                                     # Reference image configuration
        "path": "./data/ri/mnist/2.png",        # Path to the reference image
        "class_idx": 2                          # Ground truth class index of the reference image
    },

    "model": {
        "dataset": "MNIST",                     # Dataset used for training; either "MNIST" or "GTSRB"
        "hidden_layer": [6, 6, 6],              # Sizes of the hidden layers in the neural network
        "load_model": true,                     # Whether to load an existing model from disk (true) or train from scratch (false)
        "path": "./data/models/mnist",          # Path to either:
                                                #  - A directory containing a model named using the naming convention
                                                #  - A direct path to a model file (e.g., "model_i1x4x4_l6-6-6_o5.pt")
        "data_path": "./__data__",              # Location of the training data; will be downloaded if not existent
        "selected_classes": [1, 3, 5, 7, 9],    # List of classes to train on; if empty or None, all dataset classes are used
        "epochs": 50,                           # Number of training epochs
        "batch_size": 32                        # Batch size for training
    },

    "minlp": {                                  # Configuration for MINLP
        "distance": "ssim",                     # Distance metric for image similarity (only "ssim" supported atm.)
        "threshold": 0.99,                      # After repair, all images with distance > threshold to the RI must be classified correctly
        "timeout": 86400,                       # Timeout for the verification process (in seconds), e.g. 86400 = 24h; per verification run
        "tolerance": 4                          # Verification tolerance (10^(-threshold))
    },

    "repair": {
        "batch_size": 32,                       # Batch size used during the model repair process
        "penalty_increase": 1.1,                # Factor by which penalty for misclassification is increased over iterations
        "lr": 0.01                              # Learning rate used for gradient-based repair or optimisation
    }
}
```
