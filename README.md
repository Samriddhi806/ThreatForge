# ThreatForge: Adversarial-Robust IDS with GAN Augmentation and LLM-Driven Rule Generation

## What this project does

This project builds an AI-powered Intrusion Detection System (IDS) that combines three components:

1. **GAN-based augmentation** — A Conditional GAN (CTGAN) generates realistic adversarial attack traffic, used to harden a downstream classifier against evasion attempts.
2. **Explainable ML classifier** — An XGBoost model trained on CICIDS-2017 and CTU-13 network flow data, augmented with GAN-generated adversarial samples.
3. **LLM-driven rule generation** *(planned)* — Detected anomalies converted into valid Suricata IDS rules using the Claude API with MITRE ATT&CK context.

## Project status

This is an active, in-progress research project. Current state:

| Component | Status |
|---|---|
| Data loading & cleaning (CICIDS-2017, CTU-13) | Done — `eda.py`, `src/clean.py`, `src/load_ctu.py` |
| Baseline XGBoost classifier | Done — `src/baseline.py` |
| CTGAN adversarial augmentation + checkpointing | Done — `src/train_gan.py` |
| Robust classifier trained on GAN-augmented data | In progress |
| SHAP explainability per alert |  Planned |
| Claude API → Suricata rule generation (MITRE ATT&CK context) |  Planned |

## Pipeline

```
eda.py → src/clean.py → src/load_ctu.py → src/baseline.py → src/train_gan.py → (in progress: train_robust_classifier.py)
```

Each stage saves its outputs (splits, scaler, model, synthetic data) so later stages can be run independently once the earlier ones have produced their artifacts.

## Setup

```bash
git clone https://github.com/Samriddhi806/ThreatForge.git
cd ThreatForge
pip install -r requirements.txt
```

Datasets (CICIDS-2017, CTU-13) are not included in this repo due to size — place them under `CICIDS/` and `CTU/` respectively (see `.gitignore` for the expected structure).

## Running the pipeline

```bash
python eda.py              # Load and explore CICIDS-2017
python src/clean.py        # Clean, normalize, and split into train/val/test
python src/load_ctu.py     # Load and label CTU-13 for GAN training
python src/baseline.py     # Train baseline XGBoost classifier
python src/train_gan.py    # Train CTGAN, generate synthetic adversarial traffic
```