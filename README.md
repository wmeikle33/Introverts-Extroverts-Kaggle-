## Quickstart

```bash

git clone https://github.com/wmeikle33/Introverts-Extroverts.git
cd Introverts-Extroverts
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data]"
python scripts/download_data.py

## Training different models

pip install -e .
python scripts/train.py

## Predict
python scripts/predict.py --model models/model.joblib --input data/raw/test.gz --output predictions.csv

python scripts/submission.py

```

## Repo Structure

```bash

Introverts-Extroverts/
├── pyproject.toml
├── pre_commit_config.yaml
├── requirements.txt
├── requirements-dev.txt
├── src/
│   └── introverts_extroverts/
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── data.py
├── scripts/
│   ├── train.py
│   └── predict.py
├── reports/
├── notebooks/
└── tests/


```
