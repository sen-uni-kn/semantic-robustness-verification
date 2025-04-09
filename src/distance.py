# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import numpy as np
import torch
import pyscipopt as ps

class Distance:
    """
    Class providing static methods to compute similarity metrics between tensors.
    Includes both PyTorch-based and SCIP-integration versions of SSIM.
    """

    @staticmethod
    def ssim(x, y, scip_model=None, split=True):
        """
        Computes the Structural Similarity Index (SSIM) between two tensors x and y.
        
        - In default (non-SCIP) mode, returns a scalar PyTorch tensor.
        - In SCIP mode, adds constraints and variables to a SCIP model to encode SSIM as part of an optimisation problem.
        
        Args:
            x (torch.Tensor or list of SCIP variables): Input tensor or variable vector.
            y (torch.Tensor or np.ndarray): Reference tensor (must be constant when used with SCIP).
            scip_model (pyscipopt.Model, optional): SCIP model object. If provided, builds SSIM constraints.
            split (bool): If True and using SCIP, splits SSIM into luminance, contrast, and structure components.

        Returns:
            torch.Tensor or SCIP variable: SSIM value as a tensor (if `scip_model` is None),
                                           or SCIP variable representing SSIM in the constraint model.
        """
        L = 1
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        if scip_model is None:
            n = x.shape[0]
            mean_x = torch.mean(x)
            mean_y = torch.mean(y)
            var_x = torch.sum((x - mean_x) ** 2) / n
            var_y = torch.sum((y - mean_y) ** 2) / n
            covariance = torch.sum((x - mean_x) * (y - mean_y)) / n

            ssim_value = ((2 * mean_x * mean_y + C1) * (2 * covariance + C2)) / \
                         ((mean_x ** 2 + mean_y ** 2 + C1) * (var_x + var_y + C2))

            return ssim_value

        else:
            n = len(x)

            mean_x = scip_model.addVar(name="mean_x", lb=0, ub=1)
            scip_model.addCons(mean_x == ps.quicksum(x) / n)
            mean_y = np.mean(y)

            x_minus_mean = [scip_model.addVar(name=f"x_minus_mean_{i}", lb=-1, ub=1) for i in range(n)]
            for i in range(n):
                scip_model.addCons(x_minus_mean[i] == x[i] - mean_x)

            variance_x = scip_model.addVar(vtype="CONTINUOUS", name="variance_x", lb=0, ub=1)
            scip_model.addCons(variance_x == ps.quicksum(x_minus_mean[i] ** 2 for i in range(n)) / n)

            covariance = scip_model.addVar(vtype="CONTINUOUS", name="covariance", lb=-1, ub=1)
            scip_model.addCons(covariance == ps.quicksum(x_minus_mean[i] * (y[i] - mean_y) for i in range(n)) / n)

            variance_y = sum((y - mean_y) ** 2) / n

            if not split:
                numerator = scip_model.addVar(vtype="CONTINUOUS", name="numerator", lb=-(2+C1)*(2+C2), ub=(2+C1)*(2+C2))
                scip_model.addCons(numerator == (2 * mean_x * mean_y + C1) * (2 * covariance + C2))

                denominator = scip_model.addVar(vtype="CONTINUOUS", name="denominator", lb=C1 * C2, ub=(1+C1)*(1+1+C2))
                scip_model.addCons(denominator == (mean_x ** 2 + mean_y ** 2 + C1) * (variance_x + variance_y + C2))

                ssim_scip = scip_model.addVar(vtype="CONTINUOUS", name="ssim_scip", lb=-1, ub=1)
                scip_model.addCons(numerator == ssim_scip * denominator)

            else:
                # Luminance
                luminance = scip_model.addVar(vtype="CONTINUOUS", name="luminance", lb=-1, ub=1)
                scip_model.addCons(luminance == (2 * mean_x * mean_y + C1) / (mean_x ** 2 + mean_y ** 2 + C1))

                # Contrast
                contrast = scip_model.addVar(vtype="CONTINUOUS", name="contrast", lb=-1, ub=1)
                scip_model.addCons(contrast == (2 * ps.sqrt(variance_x) * np.sqrt(variance_y) + C2) / (variance_x + variance_y + C2))

                # Structure
                structure = scip_model.addVar(vtype="CONTINUOUS", name="structure", lb=-1, ub=1)
                scip_model.addCons(structure == (covariance + C2 / 2) / (ps.sqrt(variance_x) * np.sqrt(variance_y) + C2 / 2))

                # Combine
                ssim_scip = scip_model.addVar(vtype="CONTINUOUS", name="ssim_scip", lb=-1, ub=1)
                scip_model.addCons(ssim_scip == luminance * contrast * structure)

            return ssim_scip
