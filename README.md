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
> Zhao, J., Li, R., Lu, W., & Song, Y. (2021). BERTSurv: A Transformer Model for Time-to-Event Analysis.  
> [arXiv:2103.10928](https://arxiv.org/abs/2103.10928)

### Core Idea
Traditional survival models (like Cox Proportional Hazards) require structured covariates. But in healthcare, much information is in unstructured clinical notes.  
BERTSurv leverages transformer-based embeddings (BERT) to process text, then maps embeddings to survival outcomes.

- Input: Clinical notes or unstructured text  
- Encoder: BERT-based transformer (pretrained on medical/general corpora)  
- Head: Survival prediction layer using Cox loss (negative partial log-likelihood)  
- Output: Risk scores (log-risk) and survival function estimates  

### Advantages
- Handles unstructured text directly  
- Learns rich contextual embeddings  
- Outperforms traditional Cox models with hand-crafted features  
- Compatible with explainability methods (e.g., SHAP, attention maps)  

---

## Project Workflow

This notebook adapts the BERTSurv approach for a demonstration purpose, using synthetic patient notes.

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
class BERTSurv(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        risk = self.linear(cls_emb)
        return risk.squeeze(-1)

** Explainability with SHAP

| token     | impact | abs_impact |
| --------- | ------ | ---------- |
| smoker    | +0.12  | 0.12       |
| COPD      | +0.10  | 0.10       |
| sedentary | +0.08  | 0.08       |
| female    | -0.07  | 0.07       |
| active    | -0.05  | 0.05       |

# Kaplan–Meier Curves by Group

We visualize **Kaplan–Meier survival curves** stratified by patient group labels (e.g., `"Smoker"` vs `"Non-Smoker"`):

```python
groups = ["Smoker", "Non-Smoker", "Smoker"]
plot_group_km(patients, groups, tokenizer=tokenizer)

Dependencies
Install the required packages using pip:

pip install torch transformers shap lifelines matplotlib pandas


** Key Takeaways

BERTSurv adapts transformer models (like BERT) for time-to-event (survival) analysis directly from free-text data.
The notebook:
Simulates synthetic clinical data
Trains a survival model using BERTSurv
Provides risk scoring per patient
Visualizes SHAP-based explanations
Plots Kaplan–Meier survival curves
The approach is designed to be extensible to real-world EHR notes, where explainability and hazard modeling are crucial.

References
Zhao, J., Li, R., Lu, W., & Song, Y. (2021).
BERTSurv: A Transformer Model for Time-to-Event Analysis.
arXiv:2103.10928
Lifelines Documentation
https://lifelines.readthedocs.io/
SHAP for Explainability
https://shap.readthedocs.io/
HuggingFace Transformers
https://huggingface.co/transformers/


Each note is written as a 5+ line diagnostic text, for example:

