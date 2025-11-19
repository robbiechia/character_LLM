"""
In this module, we define an Optuna callback for enhanced safety during
hyperparameter tuning in Google Colab. The callback logs each completed trial,
saves the Optuna persistent database, and periodically downloads the log and
database files to the local machine to prevent data loss.

"""

from google.colab import files
import shutil
import optuna
import os
import time

def create_safety_callback(
    log_path="tuning_results.log",
    db_path="character_llm_hyperparam_tuning.db",
    drive_backup_dir='/content/drive/MyDrive/character_llm_bo_backups',
    backup_every=1,       # Backup every N trials
):
    """
    Creates an Optuna callback that ensures safety during hyperparameter tuning
    in Google Colab by periodically downloading the log and database files.

    Args:
        log_path (str): Path to the tuning results log file.
        db_path (str): Path to the Optuna SQLite database file.
        download_every (int): Interval of trials to trigger download.
    """

    os.makedirs(drive_backup_dir, exist_ok=True)

    def callback(study, trial):
        # Only backup every N trials
        if trial.number % backup_every != 0:
            return

        print(f"[Drive Backup] Saving study + logs after trial {trial.number}...")

        # Path on Drive
        drive_db = os.path.join(drive_backup_dir, "study_db.sqlite")
        drive_log = os.path.join(drive_backup_dir, "tuning_results.log")

        # 1. Sync DB from RAM → disk
        try:
            if hasattr(os, "sync"):
                os.sync()
        except:
            pass

        # 2. Copy DB + Logs into Google Drive
        try:
            shutil.copy(db_path, drive_db)
            shutil.copy(log_path, drive_log)
            print(f"[Drive Backup] Successfully saved to {drive_backup_dir}")
        except Exception as e:
            print("[Drive Backup] ERROR:", e)

    return callback