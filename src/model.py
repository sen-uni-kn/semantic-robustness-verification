# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    """
    Fully-connected feedforward neural network with configurable hidden layers.

    Args:
        input_dim (tuple): Shape of the input (e.g., (1, 28, 28) for MNIST).
        hidden_neurons (list): List of integers specifying number of neurons in each hidden layer.
        output_size (int): Number of output classes (i.e., size of the output layer).
    """
    def __init__(self, input_dim, hidden_neurons, output_size):
        super().__init__()

        activation_fn = nn.ReLU()

        self.input_dim = input_dim
        self.input_size = np.prod(input_dim)  # Flattened input dimension
        self.output_size = output_size

        # Input layer: from input to first hidden layer
        layers = [nn.Linear(self.input_size, hidden_neurons[0]), activation_fn]

        # Hidden layers
        for i in range(1, len(hidden_neurons)):
            layers.append(nn.Linear(hidden_neurons[i - 1], hidden_neurons[i]))
            layers.append(activation_fn)

        # Output layer: from last hidden layer to output
        layers.append(nn.Linear(hidden_neurons[-1], self.output_size))

        # Combine layers into a single nn.Sequential module
        self.sequential = nn.Sequential(*layers)

        # Loss and optimizer
        self.loss_fn = nn.CrossEntropyLoss() 
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Flattened input tensor.

        Returns:
            torch.Tensor: Output logits from the network.
        """
        x = self.sequential(x)
        return x
