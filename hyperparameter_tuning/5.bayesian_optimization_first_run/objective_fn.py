"""
This module defines the objective function for Bayesian optimization
to tune hyperparameters of the character LLM training process.

The set of hyperparameters to be optimized include:
- Learning rate schedule:
    - Choices: "cosine", "warmup_decay", "constant"

- Learning rate:
    - Range: 1e-4 to 3e-3 (log scale)

- Weight decay:
    - Choices: 0.0, 0.01, 0.05, 0.1
"""

import optuna
from train_wrapper import run_full_training
from bo_utils import load_config, override_config

def build_objective(train_text, output_file, config_path, global_start_time=None, global_time_limit=None):
    """
    Builds the objective function for Optuna's Bayesian optimization.

    args:
        train_text (str): The training text data.
        config_path (str): Path to the base configuration file.
    returns:
        function: The objective function to be used by Optuna.
    """

    def objective(trial):

        

        # Hyperparameters BO will search
        lr_schedule = trial.suggest_categorical(
            "lr_schedule", ["cosine", "warmup_decay", "constant"]
        )

        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 3e-3, log=True
        )

        weight_decay = trial.suggest_categorical(
            "weight_decay", [0.0, 0.01, 0.05, 0.1]
        )

        # Only tune warmup_ratio when it matters
        if lr_schedule == "warmup_decay":
            warmup_ratio = trial.suggest_float(
                "warmup_ratio", 0.02, 0.20
            )
        else:
            warmup_ratio = 0.0

        # Load base config
        config = load_config(config_path)

        # Override the BO-searched fields
        config = override_config(config, {
            "training": {
                "lr_schedule": lr_schedule,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio
            },
            "model": {
                "weight_decay": weight_decay
            }
        })

        # Run training and return final validation loss
        return run_full_training(train_text, output_file, config, trial, global_start_time=global_start_time, global_time_limit=global_time_limit)
    
    return objective
