# DriftGuard

# DriftGuard: A Hybrid Multi-Signal Framework for Early Concept Drift Detection

**A research-oriented machine learning framework for monitoring non-stationary data streams.**

## Overview

DriftGuard is a hybrid concept drift detection framework that combines multiple monitoring signals to identify changes in evolving data streams.

Traditional concept drift detectors primarily monitor changes in prediction error. However, changes in model confidence and feature–target relationships can provide additional information about evolving data distributions.

DriftGuard combines three signals:

1. **Prediction Error:** Monitors increases in prediction errors.
2. **Prediction Confidence:** Monitors reductions in model prediction confidence.
3. **Feature–Target Relationship:** Monitors changes in rolling feature–target correlations.

These signals are combined using a weighted hybrid score to generate drift-monitoring alerts.

The project evaluates the framework against conventional detectors using synthetic abrupt and gradual drift experiments, multiple random seeds, different drift speeds, and a real-world electricity dataset.

## Research Objectives

- Investigate whether multiple monitoring signals can improve early drift detection.
- Compare hybrid monitoring against conventional drift detectors.
- Evaluate detection delay and false-alarm behaviour.
- Study robustness under different gradual drift speeds.
- Examine the relationship between monitoring alerts and future predictive performance on real-world data.

## Framework Architecture

```text
Streaming Data
      |
      v
Online Machine Learning Model
      |
      +----------------------+
      |                      |
      v                      v
 Prediction              Confidence
   Error                  Monitoring
      |                      |
      +----------+-----------+
                 |
                 v
       Feature–Target
        Relationship
         Monitoring
                 |
                 v
        Signal Processing
                 |
                 v
       Weighted Hybrid Score
                 |
                 v
        Threshold-Based
             Alerts
                 |
                 v
       Evaluation and Analysis
```

## Monitoring Signals

| Signal | Purpose |
|---|---|
| Prediction error increase | Identifies degradation in predictive performance |
| Confidence drop | Identifies reductions in model confidence |
| Relationship change | Identifies changes in feature–target associations |

The current real-world configuration uses the following weights:

| Signal | Weight |
|---|---:|
| Prediction error increase | 0.60 |
| Confidence drop | 0.10 |
| Relationship change | 0.30 |

These weights were retained from the synthetic framework for the real-world evaluation.

## Datasets

### 1. Synthetic datasets

Synthetic streams are generated to evaluate:

- Abrupt concept drift
- Gradual concept drift
- Different transition speeds
- Multiple random seeds

The gradual drift generator uses two independent normally distributed features. The target relationship transitions from the first feature to the second feature.

The known drift labels are used for evaluation only, not as detector inputs.

### 2. Real-world dataset: Elec2

The project uses the Elec2 electricity market dataset available through River.

Dataset characteristics:

- 45,312 observations
- 8 input features
- Binary prediction target
- Chronological evaluation
- No random shuffling

The online baseline uses River's StandardScaler and LogisticRegression with prequential evaluation.

## Experimental Evaluation

The framework is compared with:

- ADWIN
- DDM
- EDDM
- Page-Hinkley

Evaluation metrics include:

- Detection delay
- False alarms
- Precision
- Recall
- F1-score
- Prequential accuracy
- Runtime and recovery behaviour

### Gradual Drift Results

Across five held-out seeds (47–51):

| Detector | Mean detection delay |
|---|---:|
| DriftGuard | 94.2 |
| DDM | 109.6 |
| EDDM | 190 |
| ADWIN | 199 |
| Page-Hinkley | 343.6 |

The tested DriftGuard configuration detected all five gradual drift events without false alarms in this experiment.

### Speed Robustness Results

Across 25 held-out seeds (47–71):

| Drift speed | DriftGuard delay | DDM delay |
|---|---:|---:|
| Fast | 59.92 | 61.76 |
| Medium | 94.12 | 93.88 |
| Slow | 119.16 | 142.16 |

All evaluated detectors detected the drift events across the tested seeds.

Statistical comparisons found DriftGuard significantly faster than EDDM, ADWIN, and Page-Hinkley at all three speeds. Its detection delay was not significantly different from DDM.

DriftGuard also produced some false-alarm episodes, so its performance should be considered as a trade-off between detection speed and false-alarm behaviour.

### Real-World Elec2 Evaluation

The online baseline achieved:

- Prequential accuracy: **83.74%**
- Monitoring observations: **43,913**
- Alert episodes: **136**
- Monitoring records with alerts: **47.09%**

Future non-overlapping windows beginning with an alert had an average accuracy of 81.64%, compared with 85.73% for windows beginning without an alert.

This represents an association between alerts and subsequent predictive performance, not proof that every alert corresponds to a verified concept drift event.

## Project Structure

```text
DriftGuard/
│
├── data/
│   └── processed/
│
├── results/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── hybrid_detector.py
│   ├── generate_stream.py
│   ├── generate_gradual_stream.py
│   ├── inspect_elec2.py
│   ├── train_baseline.py
│   ├── train_elec2_baseline.py
│   ├── calibrate_elec2_driftguard.py
│   ├── run_elec2_driftguard.py
│   └── ...
│
├── tests/
├── experiments/
├── notebooks/
├── paper/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DriftGuard
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Inspect the Elec2 dataset

```bash
python src/inspect_elec2.py
```

This uses River's built-in Elec2 dataset.

### Generate an abrupt drift stream

```bash
python src/generate_stream.py --seed 42 --samples 2000 --drift-point 1000
```

### Generate a gradual drift stream

```bash
python src/generate_gradual_stream.py
```

The default gradual generator uses 2,000 observations and a transition from record 800 to record 1,200.

**Important:** The real-world evaluation and several analysis scripts depend on intermediate CSV and JSON outputs. These generated files are intentionally excluded from this initial repository. Their complete execution pipeline and data preparation instructions need to be documented alongside the experiment scripts before claiming fully reproducible execution from a fresh clone.

## Research Limitations

- Experiments use a finite number of synthetic streams and random seeds.
- The current hybrid configuration is not proven optimal for all streaming environments.
- The Elec2 dataset does not provide verified drift timestamps in this evaluation.
- Real-world alerts may indicate several types of distributional or predictive changes, not necessarily concept drift.
- Higher detection sensitivity can increase false alarms.
- The results do not establish universal superiority over conventional detectors.

## Current Development Status

The core experimental framework, synthetic drift evaluations, multi-seed robustness analysis, and real-world Elec2 evaluation have been completed.

Further work includes:

- Improving experiment reproducibility.
- Completing automated tests.
- Documenting the full data preparation pipeline.
- Completing the research paper.
- Exploring adaptive thresholding and online model adaptation.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- River
- SciPy
- Matplotlib

## Research Focus

**Concept Drift Detection | Online Machine Learning | Data Stream Mining | Hybrid Monitoring | Non-Stationary Data**

---

*DriftGuard is an ongoing research and implementation project. Experimental findings are specific to the evaluated configurations and datasets.*
