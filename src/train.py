# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import os
import logging
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from model import Network
import data_utils as dutils
from tqdm import tqdm 
from pathlib import Path

def load_dataset(params):
    """
    Loads the specified dataset (MNIST or GTSRB) with preprocessing.

    Args:
        params: Parsed parameters object containing dataset info and dimensions.

    Returns:
        tuple: (train_dataset, test_dataset) as torchvision Dataset objects.
    """
    if params.model.dataset == 'GTSRB':
        train_dataset = datasets.GTSRB(
            root=params.model.data_path,
            split='train',
            transform=dutils.get_transforms(params.dimension, 'GTSRB'),
            download=True
        )
        test_dataset = datasets.GTSRB(
            root=params.model.data_path,
            split='test',
            transform=dutils.get_transforms(params.dimension, 'GTSRB'),
            download=True
        )
    elif params.model.dataset == 'MNIST':
        train_dataset = datasets.MNIST(
            root=params.model.data_path,
            train=True,
            transform=dutils.get_transforms(params.dimension, 'MNIST'),
            download=True
        )
        test_dataset = datasets.MNIST(
            root=params.model.data_path,
            train=False,
            transform=dutils.get_transforms(params.dimension, 'MNIST'),
            download=True
        )
    else:
        raise ValueError(f"Dataset {params.model.dataset} is not supported")

    return train_dataset, test_dataset

def load_or_train_model(params):
    """
    Loads a model from file or trains a new one based on provided parameters.

    Args:
        params: Parsed parameter object containing model and training specs.

    Returns:
        model: The trained or loaded PyTorch model.
        tuple: (train_loader, test_loader) for the selected or full dataset.
    """
    train_dataset, test_dataset = load_dataset(params)

    selected_classes = params.model.selected_classes
    if selected_classes:
        # Filter datasets to selected classes
        train_class_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
        test_class_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]

        subset_train_dataset = Subset(train_dataset, train_class_indices)
        subset_test_dataset = Subset(test_dataset, test_class_indices)

        # Map original labels to [0, ..., N-1]
        class_mapping = {original_label: new_label for new_label, original_label in enumerate(selected_classes)}
        subset_train_dataset = RemapLabelsDataset(subset_train_dataset, class_mapping)
        subset_test_dataset = RemapLabelsDataset(subset_test_dataset, class_mapping)
    else:
        subset_train_dataset = train_dataset
        subset_test_dataset = test_dataset
        selected_classes = sorted({label for _, label in train_dataset})

    train_loader = DataLoader(subset_train_dataset, batch_size=params.model.batch_size, shuffle=True)
    test_loader = DataLoader(subset_test_dataset, batch_size=params.model.batch_size, shuffle=False)

    # Construct model path
    hidden_layers = '-'.join(map(str, params.model.hidden_layer))
    if params.model.path.endswith(".pt"):
        model_path = params.model.path
    else:
        model_path = f"{params.model.path}/model_i{'x'.join(map(str, params.dimension))}_l{hidden_layers}_o{len(selected_classes)}.pt"

    device = torch.device("cpu")  # Enforce CPU for deterministic ops
    print(model_path)

    if not os.path.isfile(model_path) or not params.model.load_model:
        logging.info(f"training model using device {device}")
        model = Network(params.dimension, params.model.hidden_layer, len(selected_classes)).to(device)
        train(model, train_loader, params.model.epochs)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True) 
        torch.save(model.state_dict(), model_path)
    else:
        logging.info("loading existing model...")

    # Load model from disk
    model = Network(params.dimension, params.model.hidden_layer, len(selected_classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    evaluate(model, test_loader)

    return model, (train_loader, test_loader)

class RemapLabelsDataset(Dataset):
    """
    A dataset wrapper that remaps class labels using a custom label mapping.
    """
    def __init__(self, subset_dataset, class_mapping):
        self.subset_dataset = subset_dataset
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.subset_dataset)

    def __getitem__(self, idx):
        img, label = self.subset_dataset[idx]
        new_label = self.class_mapping[label]
        return img, new_label

def train(model, train_loader, epochs):
    """
    Trains the model on the training data for a number of epochs using a progress bar.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        epochs (int): Number of training epochs.
    """
    def accuracy(predictions, labels):
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float()).item()

    device = next(model.parameters()).device
    optimizer = model.optimizer
    loss_fn = model.loss_fn

    model.train()
    logging.info(f"training for {epochs} epochs...")
    
    total_batches = epochs * len(train_loader)

    with tqdm(total=total_batches, unit="batch", desc="training") as pbar:
        for epoch in range(epochs):
            running_accuracy = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_accuracy = accuracy(outputs, labels)
                running_accuracy += batch_accuracy

                pbar.update(1)

            running_accuracy /= len(train_loader)
            pbar.set_postfix(epoch=epoch + 1, accuracy=f"{running_accuracy:.2f}")

def evaluate(model, test_loader):
    """
    Evaluates the model on the test dataset and logs the accuracy.

    Args:
        model: The neural network to evaluate.
        test_loader: DataLoader containing the test set.

    Returns:
        float: Final test accuracy as a float in [0, 1].
    """
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", desc="evaluating") as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                accuracy = correct / total
                t.set_postfix(accuracy=f"{accuracy * 100:.2f}%")

    final_accuracy = 100 * correct / total
    logging.info(f"model accuracy: {final_accuracy:.2f}%")
    return final_accuracy / 100
