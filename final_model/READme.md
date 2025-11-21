# Final Model

This directory contains the culmination of our project: the fully tuned model architecture and hyperparameters trained over an extended schedule. It integrates the insights from both the ablation studies and the hyperparameter tuning phase.

The final model demonstrates how far a compact character-level transformer can be pushed under strict compute limits, and its results validate the design choices made throughout the project.

---

## Purpose of This Directory

Here we consolidate:
- the final model configuration  
- long-run training logs  
- validation and test metrics  
- plots that summarise the final training behaviour  
- any qualitative samples or outputs relevant to the model’s performance  

This directory acts as the “result” of the entire research pipeline.

---

## Structure

```
final_model/
│
├── archive/
├── final_model_results/
├── config.json
├── final_model.ipynb
├── model_architecture.py
├── setup_utils.py
└── training_utils.py
```


### **What each component contains**

- **`archive/`**  
  Previous iterations of the final model training runs, including logs and checkpoints for reference. This was due to previously running the model with incorrect configuration.

  - **`final_model_results/`**  
Plots for training/validation loss, accuracy, and test-set performance.

- **`config.json`**  
  The full architecture and hyperparameter setup used for final training.  
  This reflects the combination of:
  - rotary positional encoding  
  - GQA attention  
  - d_model 256  
  - 6 layers  
  - 4 heads  
  - sequence length 256  
  - tuned optimiser settings  

- **`final_model.ipynb/`**  
Jupyter notebook of how we trained the final model, including code snippets for loading data, model definition, training loop, and evaluation.
  
- **`model_architeture.py`**  
Python script defining the model architecture used in the final training.

- **`setup_utils.log`**
Utility script for setting up the training environment, including data loading, checkpoint saving, and preprocessing functions.

- **`training_utils.py`**
Utility functions for training, such as setting up the optimizer, learning rate scheduler, and evaluation metrics.

---

## Relationship to the Report

The final model encapsulated in this directory is the practical implementation of the theoretical and experimental findings detailed in the project report, where our goal is to maximise its performance within computational constraints.