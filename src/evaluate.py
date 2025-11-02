import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from .config import DATA_PATH, MODEL_PATH
from .data import load_data
from .features import engineer
from .train import make_target

def eval_holdout():
    df = load_data(DATA_PATH)
    Xf = engineer(df)
    y = make_target(df)
    model = load(MODEL_PATH)
    prob = model.predict_proba(Xf)[:,1]
    fpr, tpr, th = roc_curve(y, prob)
    pr, re, th2 = precision_recall_curve(y, prob)
    return {
        "auc": auc(fpr, tpr),
        "prec_rec": list(zip(pr[:50].tolist(), re[:50].tolist()))
    }

if __name__ == "__main__":
    print(eval_holdout())
