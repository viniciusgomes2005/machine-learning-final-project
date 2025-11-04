from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier


def logistic_pipe() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                class_weight="balanced",  # ajuda se houver leve desbalanceamento
                n_jobs=1,
                random_state=42,
            )),
        ]
    )


def gbm_pipe() -> CalibratedClassifierCV:
    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,   # não combine com min_data_in_leaf
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=42,
    )
    # calibração (sigmoid) para probabilidades mais bem calibradas
    return CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3, n_jobs=1)


def _best_f1_threshold(y_true: pd.Series | np.ndarray,
                       prob: pd.Series | np.ndarray) -> float:
    pr, re, th = precision_recall_curve(y_true, prob)
    # evita divisão por zero
    f1 = np.where((pr + re) > 0, 2 * pr * re / (pr + re), 0.0)
    # precision_recall_curve devolve um threshold a menos que pr/re
    # alinhar escolhendo threshold correspondente ao melhor f1 (ignorar último ponto)
    best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(th[best_idx]) if len(th) else 0.5


def metrics(y_true: pd.Series | np.ndarray,
            prob: pd.Series | np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)

    auc = roc_auc_score(y_true, prob)

    thr = _best_f1_threshold(y_true, prob)
    y_pred = (prob >= thr).astype(int)
    f1 = f1_score(y_true, y_pred)

    return {
        "auc": float(auc),
        "f1": float(f1),
    }
