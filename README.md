# Adversarial-Robust IDS with GAN Augmentation and LLM-Driven Rule Generation

## What this project does

This project builds an AI-powered Intrusion Detection System (IDS) 
that combines three novel components:

1. **GAN-Based Augmentation** — A Conditional GAN (CTGAN) generates 
   realistic adversarial attack traffic to harden the classifier 
   against evasion attempts.

2. **Explainable ML Classifier** — An XGBoost model trained on 
   GAN-augmented data, with SHAP explanations generated per alert 
   so analysts understand every detection decision.

3. **LLM-Driven Rule Generation** — Detected anomalies are 
   automatically converted into valid Suricata IDS rules using 
   the Claude API with MITRE ATT&CK context — no human analyst needed.

---

