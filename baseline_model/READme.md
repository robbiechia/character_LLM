# Baseline Model

This directory contains the initial version of our transformer architecture — a compact ≈3.2M-parameter model designed to fit comfortably within our compute budget. The baseline acts as the reference point from which all ablations and tuning studies develop.

---

## Purpose of the Baseline

Before exploring variations, we needed a stable, well-behaved model that:
- trains reliably on limited hardware (e.g., Colab T4, TPU v5e)  
- converges within 1–1.5 hours  
- provides a consistent foundation for comparisons  

The baseline therefore defines:
- the initial architecture  
- the initial training setup  
- the default hyperparameters before any tuning  
- preprocessing of the Text8 dataset  

Its outputs allow us to measure how much each later decision improves (or harms) the model.

---

## Structure of This Directory

```
baseline_model/
│
├── base_model.py
├── base_functions.py
└── base_transformer.ipynb
```

Each file provides a different part of the baseline pipeline:
- `base_model.py` defines the decoder-only transformer architecture  
- `base_functions.py` contains utility functions for batching, loss computation, etc.  
- `base_transformer.ipynb` documents the baseline training behaviour and serves as a reference notebook  

This directory represents the "starting point" of the entire project, and all later work builds on top of it.
