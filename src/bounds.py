# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import torch
import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# Create BoundedTensor Object for given input/output bounds
def _construct_bounded_tensor(in_lb: torch.Tensor, in_ub: torch.Tensor) -> BoundedTensor:
    """
    Constructs a BoundedTensor using L-infinity norm-based perturbation.

    Args:
        in_lb (torch.Tensor): Lower bound of the input.
        in_ub (torch.Tensor): Upper bound of the input.

    Returns:
        BoundedTensor: Tensor wrapped with input perturbation bounds.
    """
    input_domain = PerturbationLpNorm(x_L=in_lb, x_U=in_ub)
    return BoundedTensor(in_lb, ptb=input_domain)

# Calculate ReLU and Output bounds with auto_LiRPA
def calculate_bounds(torch_model, dimension, verbose=0):
    """
    Calculates ReLU and final output bounds for a given torch model using auto_LiRPA.

    Args:
        torch_model (torch.nn.Module): PyTorch neural network model.
        dimension (tuple): Input dimensions of the model (e.g., (1, 28, 28)).
        verbose (int): Verbosity level (0: silent, 1: summary, 2: detailed).

    Returns:
        relu_bounds (list): List of tuples containing lower and upper bounds for each ReLU layer.
        (lb, ub) (tuple): Lower and upper bounds of the final model output as flattened numpy arrays.
    """
    if verbose > 0:
        print("calculate relu and output bounds with auto_LiRPA")

    input_dim = (1, np.prod(dimension))
    lirpa_model = BoundedModule(torch_model, torch.empty(input_dim))
    t = _construct_bounded_tensor(torch.zeros(input_dim), torch.ones(input_dim))
    lb, ub = lirpa_model.compute_bounds(x=t, method='CROWN-Optimized')
    inter = lirpa_model.save_intermediate()

    lb = lb.numpy().flatten()
    ub = ub.numpy().flatten()

    if verbose > 1:
        print(f"out layer: min. lower: {min(lb):.2f}, max. upper: {max(ub):.2f}")

    relu_bounds = []
    for key, value in inter.items():
        if "/input" in key:
            vl = value[0].squeeze().detach().tolist()
            vu = value[1].squeeze().detach().tolist()
            relu_bounds.append((vl, vu))
    
    if verbose > 1:
        for i, b in enumerate(relu_bounds):
            print(f"relu layer {i}: max. lower: {max(t for t in b[0]):.2f}, min. upper: {min(t for t in b[1]):.2f}")

    return relu_bounds, (lb, ub)
