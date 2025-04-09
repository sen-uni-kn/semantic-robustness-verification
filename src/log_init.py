# Copyright (c) 2025, Jannick Sebastian Strobel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import logging
import warnings

# Setup root logger and ignore DeprecationWarnings globally
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
warnings.simplefilter("ignore", DeprecationWarning)

def config_logger(path):
    """
    Configures the global logger to log messages to both a file and the console.

    Args:
        path (str): Path to the log file where logs should be written.

    - Removes existing handlers to avoid duplicate logs.
    - Creates a file handler that writes logs to the given path.
    - Creates a console handler that prints logs to stdout.
    - Applies a unified format to both handlers.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
