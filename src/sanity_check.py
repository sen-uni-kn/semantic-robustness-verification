# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import logging
import torch
from torch.optim import Adam
from tqdm import trange
import data_utils as dutils

def f_class(model, image, correct_class):
    """
    Computes a margin-based correctness measure.

    Args:
        model (nn.Module): The neural network.
        image (torch.Tensor): Input image (1D tensor).
        correct_class (int or Tensor): Index of the correct class.

    Returns:
        torch.Tensor: relu(pred[correct_class] - max(other_classes)); 
                      >0 means correct, 0 means incorrect.
    """
    predictions = model(image).squeeze()
    max_other_classes = torch.max(torch.cat((predictions[:correct_class], predictions[correct_class+1:])))
    return torch.relu(predictions[correct_class] - max_other_classes)

def find_counterexample(model, reference_image, correct_class, threshold, distance_fn, iterations=5000, epochs=20, lr=0.01):
    """
    Attempts to find a counterexample: an input that is far (distance >= threshold) from 
    the reference image and is misclassified by the model.

    Args:
        model (nn.Module): Neural network to falsify.
        reference_image (torch.Tensor): Reference image input.
        correct_class (int): Correct class index for the reference image.
        threshold (float): Minimum required distance to qualify as a counterexample.
        distance_fn (callable): A function computing distance between two images.
        iterations (int): Number of gradient steps per epoch.
        epochs (int): Number of restarts (different random initialisations).
        lr (float): Learning rate for the Adam optimizer.

    Raises:
        Exception: If a counterexample is found (logs and saves the image).
    """
    correct_class = torch.tensor(correct_class)
    total_steps = epochs * iterations
    progress_bar = trange(total_steps, desc="Sanity check, searching for counterexamples", leave=True)

    for epoch in range(epochs):
        # Start from random image
        perturbed_image = torch.rand_like(reference_image, requires_grad=True)
        optimizer = Adam([perturbed_image], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()

            distance = distance_fn(reference_image, perturbed_image)
            class_diff = f_class(model, perturbed_image, correct_class)

            loss = -distance - class_diff  # maximise distance and misclassification
            loss.backward()
            optimizer.step()

            # Clamp pixel values to [0, 1]
            with torch.no_grad():
                perturbed_image.clamp_(min=0.0, max=1.0)

            # Check if criteria for a counterexample is met
            if distance >= threshold and class_diff == 0:
                logging.error(f"Counterexample found in epoch {epoch + 1}.")
                dutils.list_to_png(perturbed_image.detach().numpy(), "./sanity_fail.png")

                fail_image = perturbed_image.detach()
                logging.error(fail_image)
                logging.error(f"prediction: {torch.argmax(model(fail_image).squeeze())}")
                logging.error(f"distance:   {distance_fn(reference_image, perturbed_image)}")
                raise Exception("sanity check failed!")

            progress_bar.update(1)  # advance progress bar
    
    progress_bar.close()
    logging.info(f"falsification could not find counterexamples in {epochs * iterations} steps")
