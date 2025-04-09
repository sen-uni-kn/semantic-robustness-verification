# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import os
import json
import argparse
import minlp as minlp
import repair 
import train
import logging
from log_init import config_logger
import data_utils as dutils
from sanity_check import find_counterexample
from distance import Distance
import torch

# Ensure deterministic behaviour
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

def rv(params, params_dict):
    """
    Runs the verification-repair loop for a given configuration.

    Args:
        params (Namespace): Parsed parameters as an argparse.Namespace object.
        params_dict (dict): Raw parameter dictionary (for logging purposes).

    Returns:
        bool: True if the property was verified successfully, False if the time limit was reached.
    """
    dimension       = params.dimension
    ri_class_idx    = params.ri.class_idx
    distance        = params.minlp.distance

    # Construct experiment path based on configuration
    pre_ex_path = f"{params.path}/i{params.dimension[0]}x{params.dimension[1]}x{params.dimension[2]}_l" \
              + "-".join([f"{layer}" for layer in params.model.hidden_layer]) \
              + f"_o{len(params.model.selected_classes)}_{params.minlp.distance}"

    # Create unique experiment folder
    ex_path = dutils.create_next_folder(pre_ex_path)

    # Save parameters and configure logging
    dutils.write_dict_to_json(params_dict, f"{ex_path}/params.json")
    config_logger(f"{ex_path}/exp_logfile.txt")

    logging.info(f"Run experiment with distance: {distance}, input size: {dimension}, "
                 f"selected classes: {params.model.selected_classes}, repair class: {ri_class_idx}")

    # Load or train model
    torch_model, (train_loader, test_loader) = train.load_or_train_model(params)

    # Load and save reference image
    ri = dutils.load_and_transform_image(params.ri.path, params.dimension, params.model.dataset)
    dutils.list_to_png(params.dimension, ri, f"{ex_path}/reference.png")

    adversarial_examples = []
    run = 0
    ev = {"iterations": {}}

    # Start verification-repair loop
    while True:
        run += 1
        ev_verify = {}

        adv_ex = minlp.verify(params, torch_model, ri, ev_verify)

        if adv_ex == True:
            # Verification passed
            find_counterexample(torch_model, ri, params.ri.class_idx, params.minlp.threshold,
                                getattr(Distance, params.minlp.distance))
            logging.info(f"Property successfully verified after repairing {len(adversarial_examples)} CE(s)")
            try:
                os.rename(f"{ex_path}/intermediate_model.pt", f"{ex_path}/repaired_model.pt")
            except Exception:
                raise Exception("There was an error renaming the repaired model!")

            ev_verify["accuracy_after"] = train.evaluate(torch_model, test_loader)
            ev["iterations"][f"{run}"] = ev_verify
            dutils.write_dict_to_json(ev, f"{ex_path}/evaluation.json")
            return True

        elif isinstance(adv_ex, list):
            # Found counterexamples, apply repair
            old_acc = train.evaluate(torch_model, test_loader)
            dutils.list_to_png(params.dimension, adv_ex, f"{ex_path}/generated_{run}.png")
            adversarial_examples.append(adv_ex)

            correct_classes = [params.ri.class_idx] * len(adversarial_examples)
            repair.repair_model(params, torch_model, adversarial_examples, correct_classes,
                                train_loader, test_loader)

            logging.info(f"Repaired {len(adversarial_examples)} CE(s)")

            new_acc = train.evaluate(torch_model, test_loader)
            torch.save(torch_model.state_dict(), f"{ex_path}/intermediate_model.pt")

            ev_verify["accuracy_before"] = old_acc
            ev_verify["accuracy_after"] = new_acc
            ev["iterations"][f"{run}"] = ev_verify
            dutils.write_dict_to_json(ev, f"{ex_path}/evaluation.json")
            logging.info("---")

        elif not adv_ex:
            # Time limit or failure
            logging.warning("Verification failed: reached time limit.")
            ev["iterations"][f"{run}"] = ev_verify
            dutils.write_dict_to_json(ev, f"{ex_path}/evaluation.json")
            return False

        else:
            # Unexpected return
            ev["iterations"][f"{run}"] = ev_verify
            dutils.write_dict_to_json(ev, f"{ex_path}/evaluation.json")
            raise Exception("Error: return value of MINLP solver was not recognized.")


def main_run(params_path):
    """
    Entry point for the experiment runner.

    Args:
        params_path (str): Path to a JSON file containing experiment parameters.

    Returns:
        bool: True if verification was successful, False otherwise.
    """

    def dict_to_namespace(d):
        """Recursively convert a nested dictionary to an argparse.Namespace."""
        ns = argparse.Namespace()
        for key, value in d.items():
            setattr(ns, key, dict_to_namespace(value) if isinstance(value, dict) else value)
        return ns

    if not os.path.exists(params_path):
        print(f"Error: The file {params_path} does not exist.")
        return

    with open(params_path, 'r') as params_file:
        params_dict = json.load(params_file)

    return rv(dict_to_namespace(params_dict), params_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment with a specified params.json file.")
    parser.add_argument('params_path', type=str, help="Path to the params.json file")
    args = parser.parse_args()
    main_run(params_path=args.params_path)
