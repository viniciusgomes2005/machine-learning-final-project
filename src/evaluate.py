# src/evaluate.py
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from joblib import load

from .config import DATA_PATH, MODEL_PATH
from .data import load_data
from .features import engineer
from .train import make_target
from .model import gbm_pipe  # usamos o mesmo builder para avaliação limpa


def _holdout_groups(Xf: pd.DataFrame, groups: pd.Series, test_size=0.2, seed=42):
    """Cria um holdout sem vazamento por sujeito."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    ((trn_idx, tst_idx),) = splitter.split(Xf, groups=groups)
    # checagem de interseção de grupos
    g_trn = np.unique(groups.iloc[trn_idx])
    g_tst = np.unique(groups.iloc[tst_idx])
    inter = np.intersect1d(g_trn, g_tst)
    return trn_idx, tst_idx, inter.size


def eval_holdout():
    # carrega e gera features/target
    df = load_data(DATA_PATH)
    Xf_full = engineer(df)
    y_full = make_target(df)

    # saneamento mínimo (seguimos a mesma lógica do train.sanitize)
    Xf = Xf_full.copy()
    Xf.columns = Xf.columns.map(lambda c: str(c).strip().replace(" ", "_"))
    for c in Xf.columns:
        if not np.issubdtype(Xf[c].dtype, np.number):
            Xf[c] = pd.to_numeric(Xf[c], errors="coerce")
    Xf = Xf.loc[:, Xf.notna().any(axis=0)]
    const = Xf.nunique(dropna=False) <= 1
    Xf = Xf.loc[:, ~const]
    arr = Xf.to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(arr).all(axis=1)
    Xf = Xf.loc[finite_mask]
    y = y_full.loc[Xf.index]

    # grupos por sujeito (ou índice)
    if "Subject ID" in df.columns:
        groups = df.loc[Xf.index, "Subject ID"]
    else:
        groups = pd.Series(np.arange(len(Xf), dtype=int), index=Xf.index)

    # split seguro por grupos
    trn_idx, tst_idx, overlap = _holdout_groups(Xf, pd.Series(groups), test_size=0.2, seed=42)

    # treino somente no treino (modelo limpo p/ avaliação)
    model = gbm_pipe()
    model.fit(Xf.iloc[trn_idx], y.iloc[trn_idx])

    # métricas no teste
    prob = model.predict_proba(Xf.iloc[tst_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y.iloc[tst_idx], prob)
    pr, re, _ = precision_recall_curve(y.iloc[tst_idx], prob)

    out = {
        "auc": float(auc(fpr, tpr)),
        "prec_rec": list(zip(pr[:50].tolist(), re[:50].tolist())),
        "n_test": int(len(tst_idx)),
        "n_train": int(len(trn_idx)),
        "groups_overlap_in_holdout": int(overlap),  # deve ser 0
    }
    return out


if __name__ == "__main__":
    print(json.dumps(eval_holdout(), indent=2))
