import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

def logistic_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))  # sem n_jobs
    ])


def gbm_pipe():
    clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,  # use só este OU só min_data_in_leaf
        n_jobs=1,
        random_state=42,
    )
    return CalibratedClassifierCV(clf, method="sigmoid", cv=3, n_jobs=1)

    
def metrics(y_true, prob):
    y_pred = (prob >= 0.85).astype(int)
    return {
        "auc": roc_auc_score(y_true, prob),
        "f1": f1_score(y_true, y_pred)
    }
