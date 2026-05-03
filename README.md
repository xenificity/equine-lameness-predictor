# 🐴 Mizzou Lameness Predictor

A machine learning web app that predicts the **anatomical location of equine lameness** from inertial sensor gait data and clinical records.

**Live demo →** https://mizzou-lameness-predictor.streamlit.app

---

## Overview

This tool combines three data modalities to localize lameness to one of four anatomical zones:

| Predicted Class | Region | Training Prevalence |
|---|---|---|
| **Distal** | Foot, pastern, fetlock | 42.9% |
| **Middle** | Carpus (knee), tarsus (hock), stifle | 41.5% |
| **Proximal** | Shoulder, elbow, hip, upper hindlimb | 12.4% |
| **Non-Limb** | Back, sacroiliac, cervical/thoracic/lumbar spine | 3.2% |

### Model Performance

| Metric | Value |
|---|---|
| Overall accuracy | **94.4%** |
| Distal F1 | 0.97 |
| Middle F1 | 0.94 |
| Proximal F1 | 0.95 |
| Non-limb recall | 85% |
| Training set | 8,454 stride observations / 514 horses |

---

## Features

- **16 inertial sensor metrics** from head-mounted and pelvic IMU systems
- **17 structured clinical record fields** (management, exam findings, diagnostics)
- **20 anatomical keyword flags** from veterinary narrative text (regex)
- **60 TF-IDF bigram features** from combined history + reason text
- **96 case-level gait aggregation features** (16 sensors × 6 statistics)

---

## Model Architecture

```
Input (209 features)
       │
       ▼
┌──────────────────────────────┐
│  Stage 1: LightGBM (binary)  │  ── Non-limb vs Limb
│  98.1% binary accuracy       │
└───────────────┬──────────────┘
                │ Limb cases only
                ▼
┌──────────────────────────────────────────────────┐
│  Stage 2: Stacking Ensemble                      │
│  ┌────────────┐ ┌──────────┐ ┌──────────────┐   │
│  │  LightGBM  │ │  XGBoost │ │  Extra Trees │   │
│  └─────┬──────┘ └────┬─────┘ └──────┬───────┘   │
│        └─────────────┴──────────────┘            │
│                       │ OOF probabilities         │
│              ┌────────▼────────┐                 │
│              │ Logistic Regr.  │  (meta-learner) │
│              └─────────────────┘                 │
└──────────────────────────────────────────────────┘
                       │
                       ▼
         distal / middle / proximal / nonlimb
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/equine-lameness-predictor
cd equine-lameness-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The pre-trained model (`models/pipeline.pkl`) is included in the repository (13 MB, joblib-compressed).

---

## Deploying to Streamlit Community Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy** — your app gets a public `*.streamlit.app` URL

---

## Project Structure

```
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── models/
│   └── pipeline.pkl          # Trained model (joblib-compressed, 13 MB)
└── .streamlit/
    └── config.toml           # App theme and server config
```

---

## Research

This app accompanies a conference presentation documenting all eight model iterations
from 49.3% baseline to 94.4% final accuracy.

| Experiment | Configuration | Accuracy |
|---|---|---|
| 1 | Baseline LightGBM (sensor only) | 49.3% |
| 2 | + class_weight=balanced | 54.7% |
| 3 | + Medical record features | 58.3% |
| 4 | + TF-IDF text + anatomical flags | 68.5% |
| 5 | Two-stage architecture | 78.3% |
| 6 | + Case-level gait aggregations | 85.2% |
| 7 | + Optuna Bayesian HPO | 85.2% |
| 8 | + Stacking ensemble | **94.4%** |

---

## Disclaimer

For research and clinical decision support use only.
Not a substitute for veterinary examination or clinical judgment.

**MU Veterinary Health Center · University of Missouri**
