# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from torchvision import transforms
import torch
from PIL import Image
import math
import os
import json
from ilock import ILock

def get_transforms(dim, dataset):
    """
    Returns a composed set of image transformations based on the input dimensions and dataset name.

    Args:
        dim (tuple): Input dimensions (channels, height, width).
        dataset (str): Dataset name ("MNIST" or "GTSRB").

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    if dataset == "GTSRB":
        return transforms.Compose([
            transforms.Resize(dim[1:]),  
            transforms.ToTensor(),  
            transforms.Lambda(lambda x: x.view(-1))])
    elif dataset == "MNIST":
        return transforms.Compose([
            transforms.Resize(dim[1:]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))])

def load_and_transform_image(image_path, dim, dataset):
    """
    Loads an image and applies the appropriate transformations.

    Args:
        image_path (str): Path to the image file.
        dim (tuple): Target input dimensions.
        dataset (str): Dataset name ("MNIST" or "GTSRB").

    Returns:
        torch.Tensor: Transformed image as a flattened tensor.
    """
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return get_transforms(dim, dataset)(image)

def list_to_png(dim, pixel_list, output_path):
    """
    Converts a flat pixel list into a PNG image and saves it.

    Args:
        dim (tuple): Image dimensions (channels, height, width).
        pixel_list (list): Flattened list of pixel values.
        output_path (str): Path to save the resulting PNG file.
    """
    image_tensor = torch.as_tensor(pixel_list)
    l = int(math.sqrt(len(pixel_list) / dim[0]))

    if l * l * dim[0] != len(pixel_list):
        raise ValueError("The pixel list length is not compatible with a square image.")

    dim = (dim[0], l, l)
    image_tensor = image_tensor.view(*dim)

    pil_image = transforms.ToPILImage()(image_tensor)
    pil_image.save(output_path, format='PNG')

def create_next_folder(base_folder, directory="."):
    """
    Creates a new uniquely indexed subfolder within the given directory.

    Args:
        base_folder (str): Base name for the folder (e.g., 'experiment').
        directory (str): Parent directory where the folder will be created.

    Returns:
        str: Name of the newly created folder.
    """
    with ILock('create_folder_lock'):
        index = 1
        next_folder = f"{base_folder}_{index:03d}"
        while os.path.exists(os.path.join(directory, next_folder)):
            index += 1
            next_folder = f"{base_folder}_{index:03d}"
        os.makedirs(os.path.join(directory, next_folder))
        return next_folder

def write_dict_to_json(data, file_path, indent=4, ensure_ascii=True, sort_keys=False):
    """
    Writes a dictionary to a JSON file, creating the directory if needed.

    Args:
        data (dict): Dictionary to be written.
        file_path (str): Path to the JSON file to write.
        indent (int, optional): Indentation level for pretty printing.
        ensure_ascii (bool, optional): Escape non-ASCII characters if True.
        sort_keys (bool, optional): Sort keys in output JSON if True.

    Raises:
        TypeError: If input data is not a dictionary or can't be serialized.
        OSError: If directory creation or file writing fails.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected data to be a dict, but got {type(data).__name__}")

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            raise OSError(f"Failed to create directory {directory}: {e}")

    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        print(f"Dictionary successfully written to {file_path}")
    except TypeError as te:
        raise TypeError(f"Failed to serialize data to JSON: {te}")
    except OSError as oe:
        raise OSError(f"Failed to write to file {file_path}: {oe}")
