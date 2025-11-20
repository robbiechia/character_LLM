"""
In this file, we provide utility functions for setting up and managing the training environment
of the character-level language model. This includes functions for:

1. Logging:
    - initialize_training_log: Create and initialize the training log file.
    - initialize_validation_log: Create and initialize the validation log file.
    - initialize_test_log: Create and initialize the test log file.
    - setup: Set up logging files for training, validation, and testing.
    - update_training_log: Append new entries to the training log.
    - update_validation_log: Append new entries to the validation log.
    - update_test_log: Append new entries to the test log.

2. Checkpointing:
    - save_checkpoint: Save model parameters and optimizer state as a checkpoint.
    - load_checkpoint: Load model parameters and optimizer state from a checkpoint.

3. Config Management:
    - load_config: Load experiment configuration from a JSON file.

4. Dataset Management:
    - load_dataset: Load training and testing datasets from text files.
    - split_train_val: Split data into training and validation sets.

5. Save to Google Drive:
    - save_to_drive: Save local files to a Google Drive backup directory.
"""
import json
import os
import pickle
import shutil

# ------- #
# Logging #
# ------- #

def initialize_training_log(log_path):
    """
    Initializes the training log file with a header.
    
    Args:
        log_path (str): Path to the log file.
    """

    with open(log_path, 'w') as log_file:
        log_file.write("step, train_loss, train_time, train_acc, last_char_acc")

    print(f"[initialize_training_log] Initialized training log file at {log_path}")

def initialize_validation_log(log_path):
    """
    Initializes the validation log file with a header.
    
    Args:
        log_path (str): Path to the log file.
    """

    with open(log_path, 'w') as log_file:
        log_file.write("step, val_loss, val_time, val_acc, last_char_val_acc")

    print(f"[initialize_validation_log] Initialized validation log file at {log_path}")

def initialize_test_log(log_path):
    """
    Initializes the test log file with a header.
    
    Args:
        log_path (str): Path to the log file.
    """

    with open(log_path, 'w') as log_file:
        log_file.write("test_loss, test_acc, last_char_test_acc")

    print(f"[initialize_test_log] Initialized test log file at {log_path}")

def setup(training_log_file, validation_log_file, test_log_file):
    """
    Sets up the logging files for training, validation, and testing.

    Args:
        training_log_file (str): Path to the training log file.
        validation_log_file (str): Path to the validation log file.
    """

    if not os.path.exists(training_log_file):
        initialize_training_log(training_log_file)

    if not os.path.exists(validation_log_file):
        initialize_validation_log(validation_log_file)

    if not os.path.exists(test_log_file):
        initialize_test_log(test_log_file)

def update_training_log(log_path, step, train_loss, train_time, train_acc, last_char_acc):
    """
    Appends a new entry to the training log file.

    Args:
        log_path (str): Path to the log file.
        step (int): Current training step.
        train_loss (float): Training loss.
        train_time (float): Time taken for training.
        train_acc (float): Training accuracy.
        last_char_acc (float): Last character accuracy on training set.
    """

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{step}, {train_loss:.4f}, {train_time:.2f}, {train_acc:.4f}, {last_char_acc:.4f}")
    
def update_validation_log(log_path, step, val_loss, val_time, val_acc, last_char_val_acc):
    """
    Appends a new entry to the training log file.

    Args:
        log_path (str): Path to the log file.
        step (int): Current training step.
        val_loss (float): Validation loss.
        val_time (float): Time taken for validation.
        val_acc (float): Validation accuracy.
        last_char_val_acc (float): Last character accuracy on validation set.
    """

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{step}, {val_loss:.4f}, {val_time:.2f}, {val_acc:.4f}, {last_char_val_acc:.4f}")

def update_test_log(log_path, test_loss, test_acc, last_char_test_acc):
    """
    Appends a new entry to the test log file.

    Args:
        log_path (str): Path to the log file.
        test_loss (float): Test loss.
        test_acc (float): Test accuracy.
        last_char_test_acc (float): Last character accuracy on test set.
    """

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{test_loss:.4f}, {test_acc:.4f}, {last_char_test_acc:.4f}")  
    
# ------------- #
# Checkpointing #
# ------------- #

def save_checkpoint(checkpoint_path, params, constants, opt_state, step, time_elapsed):
    """
    Saves the model parameters and optimizer state as a checkpoint.

    Args:
        checkpoint_path (str): Path to save the checkpoint.
        params (dict): Model parameters.
        opt_state (dict): Optimizer state.
        step (int): Current training step.
    """

    checkpoint = {
        'step': step,
        'params': params,
        'constants': constants,
        'opt_state': opt_state,
        'time_elapsed': time_elapsed
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    

    print(f"[save_checkpoint] Saved checkpoint at step {step} to {checkpoint_path}")

def load_checkpoint(checkpoint_path, params, constants, opt_state):
    """
    Loads the model parameters and optimizer state from a checkpoint if it exists.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
    Returns:
        params (dict): Model parameters.
        opt_state (dict): Optimizer state.
        step (int): Training step at which the checkpoint was saved.
        time_elapsed (float): Time elapsed until the checkpoint was saved.
    """

    if not os.path.exists(checkpoint_path):
        print(f"[load_checkpoint] No checkpoint found at {checkpoint_path}")
        print(f"[load_checkpoint] Starting training as per normal.")

        return params, opt_state, constants, 0

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"[load_checkpoint] Loaded checkpoint from {checkpoint_path} at step {checkpoint['step']}")
    print(f"[load_checkpoint] Resuming training from step {checkpoint['step']}")
    print(f"[load_checkpoint] Time elapsed until checkpoint: {checkpoint['time_elapsed']:.2f} seconds")
    
    return checkpoint["params"], checkpoint["opt_state"], checkpoint["constants"], checkpoint["step"]

# ----------------- #
# Config Management #
# ----------------- #

def load_config(config_path):
    """
    Loads experiment configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.
    Returns:
        config (dict): Experiment configuration.
    """
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"[load_config] Loaded configuration from {config_path}")

    name = config["name"]
    description = config["description"]
    seed = config["seed"]
    model = config["model"]
    training = config["training"]
    throughput = config["throughput"]
    
    return name, description, seed, model, training, throughput

# ------------------ #
# Dataset Management #
# ------------------ #

def load_dataset(train_file, test_file):
    """
    Loads training and testing datasets from text files.

    Args:
        train_file (str): Path to the training text file.
        test_file (str): Path to the testing text file.
    Returns:
        train_text (str): Training text data.
        test_text (str): Testing text data.
    """
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    print(f"[load_dataset] Loaded training text from {train_file}. Length: {len(train_text) :,} characters.")

    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()
    print(f"\n[load_dataset] Loaded testing text from {test_file}. Length: {len(test_text) :,} characters.")

    # Inspect first 100 characters of training text
    print(f"\n[load_dataset] First 100 characters of training text:\n{train_text[:100]}")

    # Inspect first 100 characters of testing text
    print(f"\n[load_dataset] First 100 characters of testing text:\n{test_text[:100]}")

    # Create character to integer and integer to character mappings
    unique_chars = sorted(set(train_text))
    chars_to_int = {ch: i for i, ch in enumerate(unique_chars)} # Character to integer mapping
    int_to_chars = {i: ch for i, ch in enumerate(unique_chars)} # Integer to character mapping
    print(f"\n[load_dataset] Created character mappings. Vocabulary size: {len(unique_chars)}")

    return train_text, test_text, chars_to_int, int_to_chars

def split_train_val(data, val_fraction=0.1):
    """
    Splits data into training and validation sets.
    We ensure that the split is done at the last whitespace before the split point.
    If no whitespace is found, we split at the exact point.
    
    Args:
		data: string, the full text data
        val_fraction: fraction of data to use for validation
        
    Returns:
		train_data: string, training data
        val_data: string, validation data
    """
    
    split_idx = int(len(data) * (1 - val_fraction)) # Initial split index based on val_fraction
    last_space_idx = data.rfind(' ', 0, split_idx) # Find last whitespace before split_idx
    
    if last_space_idx != -1: # If a whitespace is found, split there
        split_idx = last_space_idx
        
    train_data = data[:split_idx] # Training data up to split_idx
    val_data = data[split_idx:] # Validation data from split_idx to end

    print(f"[split_train_val] Training text length: {len(train_data) :,} characters.")
    print(f"[split_train_val] Validation text length: {len(val_data) :,} characters.")

    return train_data, val_data

# -------------------- #
# Save to Google Drive #
# -------------------- #

def save_to_drive(file_name, drive_backup_dir):
    """
    Saves a local file to Google Drive backup directory.

    Args:
        local_path (str): Path to the local file.
        drive_backup_dir (str): Path to the Google Drive backup directory.
    """

    # Path on Drive
    drive_path = os.path.join(drive_backup_dir, file_name)

    # Copy file into Google Drive
    try:
        shutil.copy(file_name, drive_path)
        print(f"[save_to_drive] Successfully saved {file_name} to {drive_backup_dir}")
    except Exception as e:
        print("[save_to_drive] ERROR:", e)