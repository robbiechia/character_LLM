# Character-Level Transformer Language Model

This repository documents our journey in building a compact, efficient character-level Transformer model trained on the Text8 dataset. Rather than scaling towards billions of parameters, our goal was to understand — in a controlled, compute-constrained environment — how architectural decisions and training choices shape model performance. The project is therefore organised around experimentation: starting from a simple baseline, we progressively refine the model through systematic ablations and targeted hyperparameter tuning.

The end result is a coherent pipeline that reflects both the practical limitations we worked under and the research-driven process that guided each decision.

---

## Project Overview

Our central task is **next-character prediction** on the Text8 corpus — a 100M-character cleaned Wikipedia dump containing only lowercase letters and spaces. We train a **decoder-only Transformer**, chosen because autoregressive character modelling does not require an encoder. The model learns the distribution:

$P_\theta(x_{t+1} \mid x_{t-L+1}, \dots, x_t)$

over a context window of length \(L\). This setup allows us to study the role of context length, model size, attention configuration, and other components with clarity.

---

## Repository Structure

The repository is organised to reflect the logical progression of our project:

```
character_LLM
├── baseline_model/
│   └── Our starting reference model (≈3.2M parameters)   
│
├── ablations/
│   └── Systematic tests of architectural and design choices
│
├── hyperparameter_tuning/
│   └── Fine-grained parameter search, including Bayesian optimisation
│
└── final_model/
    └── Final configuration, extended training, and consolidated results
```


Each directory contains its own detailed README explaining its purpose and internal structure.

---

## Thought Process and Methodology

This project follows a structured research progression:

1. **Establish a Baseline**  
   A small, stable model that fits comfortably within the compute limits of Colab T4/Kaggle GPUs.

2. **Architectural Ablations**  
   Using a one-factor-at-a-time approach, we vary:
   - loss formulation  
   - optimiser  
   - positional encoding  
   - attention mechanism  
   - model width, depth, and number of heads  
   - sequence length  

   At each stage, the best-performing configuration becomes the new default.

3. **Hyperparameter Tuning**  
   Once the architecture stabilised, we tuned:
   - batch size  
   - gradient clipping  
   - dropout  
   - label smoothing  
   - learning rate schedule  
   - learning rate
   - weight decay  

   We used Bayesian Optimisation for learning rate schedule, learning rate and weight decay, guided by the need to operate under strict time limits.

4. **Final Model Training**  
   The best architecture and hyperparameters are combined and trained deeply to evaluate generalisation on the held-out test set.

All findings, insights, and methodological reflections are detailed in the accompanying project report.

---

## Report

A full analysis of the experiments, motivations, and results can be found in the project report included in this repository.
