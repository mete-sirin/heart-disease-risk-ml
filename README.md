# Heart Disease Risk Prediction

ANN course assignment. Binary classification of heart disease presence (0/1) from clinical features using the UCI Cleveland Heart Disease dataset.

Compares a Logistic Regression baseline against two ANN architectures. Priority metric is recall — minimising missed diagnoses matters more than avoiding false alarms.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Runs the full pipeline: EDA → preprocessing → baseline → ANN training → results summary. Output figures are saved to `outputs/eda/`.

## Stack

Python 3.11 · scikit-learn · TensorFlow/Keras · pandas · matplotlib · seaborn
