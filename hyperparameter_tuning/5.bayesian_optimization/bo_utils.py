"""
This module provides utility functions to load and override configurations
for hyperparameter tuning using Bayesian optimization.
"""
import json
import copy

def load_config(path="./config.json"):
    with open(path, "r") as f:
        return json.load(f)

def override_config(base_config, override_dict):
    """
    Returns a deep-copied config with override_dict values applied.
    e.g.
        override_config(cfg, {"training": {"learning_rate": 0.001}})

    args:
        base_config (dict): The base configuration dictionary.
        override_dict (dict): A dictionary containing parameters to override.
    returns:
        dict: The overridden configuration dictionary.
    """
    config = copy.deepcopy(base_config)
    for section, params in override_dict.items():
        if section in config:
            config[section].update(params)
    return config

def initialize_training_log():
    """
    Initializes the tuning results log file with a header.
    
    Args:
        log_path (str): Path to the log file.
    """

    with open("tuning_results.log", 'w') as log_file:
        log_file.write("Beginning of Hyperparameter Tuning Results\n")

    print(f"[initialize_training_log] Initialized tuning results log file.")