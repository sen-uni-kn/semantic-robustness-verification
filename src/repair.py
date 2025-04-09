# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def f_sat(torch_model, image, correct_class):
    """
    Computes the satisfaction function for a classification decision.
    
    Args:
        torch_model (nn.Module): The neural network model.
        image (torch.Tensor): Input image tensor (flattened).
        correct_class (int): Index of the correct class.

    Returns:
        torch.Tensor: Difference between correct class score and max of other scores.
                      Negative if misclassified.
    """
    output = torch_model(image.unsqueeze(0))  # Add batch dimension
    predicted_scores = output[0]
    correct_score = predicted_scores[correct_class]
    max_other_score = torch.max(torch.cat((predicted_scores[:correct_class],
                                           predicted_scores[correct_class+1:])))
    return correct_score - max_other_score  # Negative means misclassified

def penalized_loss(torch_model, images, correct_classes, penalty_weights):
    """
    Computes the penalised loss over all counterexamples.

    Args:
        torch_model (nn.Module): The neural network.
        images (list of torch.Tensor): Counterexample inputs.
        correct_classes (list of int): True class labels for each input.
        penalty_weights (list of float): Per-input penalty weights.

    Returns:
        torch.Tensor: Total penalised loss (sum over all counterexamples).
    """
    loss = 0
    for i, image in enumerate(images):
        f_sat_val = f_sat(torch_model, image, correct_classes[i])
        loss += penalty_weights[i] * F.relu(-f_sat_val)  # Penalise misclassifications
    return loss

def repair_model(params, torch_model, images, correct_classes, train_loader, test_loader):
    """
    Applies a repair loop to update the model until all given counterexamples are correctly classified.

    Args:
        params (Namespace): Parameters with repair settings (lr, penalty_increase).
        torch_model (nn.Module): The model to be repaired (in-place).
        images (list): List of flattened counterexample images (numpy or torch).
        correct_classes (list): Corresponding class labels for the counterexamples.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for evaluation (not used here).

    Returns:
        torch.nn.Module: The repaired model.
    """
    def accuracy(predictions, labels):
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float()).item()
    
    # Convert counterexamples to tensors if needed
    images = [torch.tensor(image) for image in images]

    # Initial penalty weights
    penalty_weights = [1.0] * len(images)

    optimizer = optim.Adam(torch_model.parameters(), lr=params.repair.lr)
    original_loss_fn = nn.CrossEntropyLoss()

    iteration = 0

    with tqdm(total=len(train_loader), unit="iter", desc="repair") as pbar:
        while True:
            pbar.n = 0
            pbar.last_print_n = 0
            pbar.refresh()

            iteration += 1
            torch_model.train()
            optimizer.zero_grad()
            running_accuracy = 0.0

            for inputs, labels in train_loader:
                optimizer.zero_grad()

                outputs = torch_model(inputs)
                loss = original_loss_fn(outputs, labels)

                # Add penalty term to loss for misclassified counterexamples
                penalty_loss = penalized_loss(torch_model, images, correct_classes, penalty_weights)
                total_loss = loss + penalty_loss

                total_loss.backward()
                optimizer.step()

                batch_accuracy = accuracy(outputs, labels)
                running_accuracy += batch_accuracy
                
                pbar.update(1)
        
            # Check which counterexamples are now classified correctly
            repaired_count = 0
            with torch.no_grad():
                for i, image in enumerate(images):
                    if f_sat(torch_model, image, correct_classes[i]) >= 0:
                        repaired_count += 1
                    else:
                        penalty_weights[i] *= params.repair.penalty_increase

            running_accuracy /= len(train_loader)
            pbar.set_postfix(run=iteration, accuracy=f"{running_accuracy:.2f}", repaired=f" {repaired_count:3d}/{len(images)}")

            if repaired_count == len(images):
                break

    return torch_model
