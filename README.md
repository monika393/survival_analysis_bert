# Survival Analysis with BERTSurv

This project demonstrates how to apply **BERTSurv** — a transformer-based deep survival model — to clinical free-text data for predicting survival outcomes, analyzing hazards, and explaining predictions using SHAP.

The notebook integrates:  
- Synthetic patient note generation with risk factors (age, sex, smoking, comorbidities, etc.)  
- Survival model training/inference using BERTSurv  
- Explainability with SHAP token attribution  
- Illustrative survival curves using Kaplan–Meier  

---

## Background: What is BERTSurv?

**BERTSurv** was proposed in the paper:  
> Zhao, J., Li, R., Lu, W., & Song, Y. (2021).  
> BERTSurv: A Transformer Model for Time-to-Event Analysis.  
> [arXiv:2103.10928](https://arxiv.org/abs/2103.10928)

### Core Idea

Traditional survival models (like Cox Proportional Hazards) require structured covariates. But in healthcare, much information is in unstructured clinical notes.  
BERTSurv leverages transformer-based embeddings (BERT) to process text, then maps embeddings to survival outcomes.

- **Input:** Clinical notes or unstructured text  
- **Encoder:** BERT-based transformer (pretrained on medical/general corpora)  
- **Head:** Survival prediction layer using Cox loss (negative partial log-likelihood)  
- **Output:** Risk scores (log-risk) and survival function estimates  

### Advantages

- Handles unstructured text directly  
- Learns rich contextual embeddings  
- Outperforms traditional Cox models with hand-crafted features  
- Compatible with explainability methods (e.g., SHAP, attention maps)  

---

## Project Workflow

This notebook adapts the BERTSurv approach for demonstration purposes, using synthetic patient notes.

### 1. Synthetic Data Generation

We simulate clinical notes with key attributes:

- Demographics: Age (40–100), Sex (Male/Female)  
- Lifestyle: Smoker/Non-smoker, Activity level  
- Comorbidities: COPD, Diabetes, Hypertension  
- Cancer stage: None/I/II/III  

We then assign latent risk scores via hand-crafted coefficients (e.g., smoking and COPD increase risk, female sex decreases risk). Survival times are simulated from an exponential distribution with censoring.

---

### 2. Model Definition: BERTSurv

We define a PyTorch-based BERTSurv class:

- Loads a pretrained HuggingFace BERT model  
- Adds a linear survival head producing log-risk scores  
- Trains using Cox partial likelihood loss  

```python
import torch.nn as nn
from transformers import AutoModel

class BERTSurv(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        risk = self.linear(cls_emb)
        return risk.squeeze(-1)
```python
