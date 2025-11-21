# Hyperparameter Tuning

After establishing the model architecture through ablations, this directory explores the finer details of training dynamics. Here we tune parameters that govern optimisation behaviour rather than structural properties of the model. The goal is to extract as much performance as possible from our fixed architecture under real-world compute constraints.

---

## Structure of This Directory

```
hyperparameter_tuning/
│
├── 1_batch_size/
├── 2_dropout/
├── 3_label_smoothing/
├── 4_gradient_clipping/
├── 5.bayesian_optimization_fourth_run/
└── experiment_setup/
└── experiment_setup_kaggle/
└── archive_runs/
```


### **Individual tuning folders (e.g., `1_batch_size/`)**
Each folder contains:
- multiple configurations for that parameter  
- training outputs and plots  
- a small notebook or written summary of what we observed  

These experiments intentionally mirror the structure of the ablations: isolate one variable, observe its effect, and adopt the best setting as the new default.

---

## Thought Process Behind This Section

Hyperparameters such as dropout or gradient clipping often interact subtly with the model’s optimisation landscape. However, because our architecture is small and trained on a large dataset (100M characters), many regularisation mechanisms ended up being unnecessary — a pattern consistent with literature on models of this scale.

To tune the more sensitive parameters — chiefly the learning rate schedule and weight decay — we used **Bayesian Optimisation (Optuna)**. This allowed us to explore a large search space efficiently under time limits.

The `5.bayesian_optimization_fourth_run/` folder contains:
- full Optuna study logs  
- Jupyter notebooks analysing the results
- tuning results log
- visualisations including:
- importance analyses  
- slices/contours  
- parallel coordinate plots  

These visualisations illustrate the interaction between hyperparameters and guided the selection of the final configuration.

---

## `experiment_setup/`

Shared utility code for running tuning experiments. Similar to the ablation version, it ensures consistent configuration and provides helper functions used by multiple tuning categories.

## `experiment_setup_kaggle/`

Shared utility code for running tuning experiments on Kaggle’s platform. It includes adaptations for the environment and resource constraints specific to Kaggle.

## `archive_runs/`

Contains older tuning experiments that were superseded by later runs. These are kept for reference but are not part of the main analysis. These runs include initial attempts that utilised incorrect settings.
