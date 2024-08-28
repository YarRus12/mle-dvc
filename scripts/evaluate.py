# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os


# оценка качества модели
def evaluate_model():

    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    data = pd.read_csv('data/initial_data.csv')
    pipeline = joblib.load('models/fitted_model.pkl')

    cv_strategy = StratifiedKFold(n_splits=5)
    cv_res = cross_validate(
        pipeline,
        data,
        data['target'],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc']
    )
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(cv_res, fd)

if __name__ == '__main__':
    evaluate_model()
