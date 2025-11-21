# Ablation Studies

This directory contains the backbone of our experimental work: a structured series of ablations that isolate the effect of individual architectural and training components. Because we operated under strict GPU time limits, our ablations follow a purposeful “one-factor-at-a-time” design. Each experiment varies a single component while holding all others fixed, allowing us to form clean comparisons and clear conclusions.

The ablation pipeline mirrors the progression documented in our report: we begin with foundational architectural elements, then move towards model scaling, and finally evaluate sequence length.

---

## Structure of the Ablations Directory

Each top-level folder corresponds to a *category* of experiments:

```
ablations/
│
├── 1.loss_function/
├── 2.optimizer/
├── 3.positional_encoding/
├── 4.attention_mechanism/
├── 5.model_params/
├── 6.seq_len/
└── experiment_setup/
└── ablation_sumary.ipynb
```

### **Each Category folders (e.g., `1.loss_function/`)**

Each category folder contains:
- Subfolders representing the exact variants tested  
- Outputs such as logs, plots, and configuration snapshots, and the notebook used to run the experiment for that category
- A `results_comparison.ipynb` notebook reviewing the differences in performance across the variants  

Each folder essentially captures a mini-experiment, documenting:
- what was tested  
- how it was varied  
- what the results looked like  
- which design choice was ultimately selected  

This structure reflects the staged progression of our project: each category’s findings informed the default configuration used in the next.

---

## Flow of Experiments

The categories follow the chronological order in which we conducted our studies:

1. **Foundational architectural components**  
   - How should the model compute loss?  
   - Which optimiser is most stable under limited compute?  
   - Do different positional encodings matter at this scale?  
   - Does MHA/MQA/GQA make a difference for small models?

2. **Model scaling**  
   After fixing the above, we compared different:
   - numbers of layers  
   - model dimensions  
   - numbers of attention heads  

   These help determine the most compute-optimal architecture.

3. **Sequence length**  
   Finally, we evaluated how extending the context window affects convergence and performance.

The overall trajectory — and how each category’s conclusions build on one another — can be seen clearly in the summary notebooks.

---

## `experiment_setup/`

This folder contains helper scripts and configuration templates shared across ablation experiments. These files act as the “infrastructure” of our experiments but do not contain results. They ensure consistent setup across categories and help maintain reproducibility.

## `ablation_sumary.ipynb`

This script generates the validation loss curves of all ablation experiments after its configuration is adopted into the final model. It provides a consolidated view of how each design choice impacted performance over the course of training.

