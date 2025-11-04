# src/train.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from joblib import dump, parallel_backend

from .config import DATA_PATH, MODEL_PATH
from .data import load_data
from .features import engineer
from .model import logistic_pipe, gbm_pipe, metrics

# reduzir threads de BLAS/NumPy em ambientes Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def make_target(df: pd.DataFrame) -> pd.Series:
    # alvo: ineficiência do sono (< 0.85)
    return (df["Sleep efficiency"] < 0.85).astype(int)


def _sanitize_features_targets(Xf: pd.DataFrame, y: pd.Series):
    Xf = Xf.copy()

    # nomes padronizados
    Xf.columns = Xf.columns.map(lambda c: str(c).strip().replace(" ", "_"))

    # força numérico
    for col in Xf.columns:
        if not np.issubdtype(Xf[col].dtype, np.number):
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce")

    # remove colunas 100% NaN
    Xf = Xf.loc[:, Xf.notna().any(axis=0)]

    # remove colunas constantes
    const_mask = Xf.nunique(dropna=False) <= 1
    if const_mask.any():
        Xf = Xf.loc[:, ~const_mask]

    # descarta linhas com não finitos
    arr = Xf.to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(arr).all(axis=1)
    Xf = Xf.loc[finite_mask]
    y = y.loc[Xf.index]

    # dropa NaN remanescente
    Xf = Xf.dropna(axis=0)
    y = y.loc[Xf.index]

    return Xf, y


def _adaptive_gkf(groups: pd.Series, max_splits: int = 5) -> GroupKFold:
    n_groups = int(pd.Series(groups).nunique())
    n_splits = max(2, min(max_splits, n_groups))
    return GroupKFold(n_splits=n_splits)


def _cv_scores(name, builder, Xf, y, groups, gkf):
    fold_scores = []
    with parallel_backend("threading", n_jobs=1):
        for trn_idx, val_idx in gkf.split(Xf, y, groups):
            model = builder()
            X_tr, y_tr = Xf.iloc[trn_idx], y.iloc[trn_idx]
            X_va, y_va = Xf.iloc[val_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_va)[:, 1]
            fold_scores.append(metrics(y_va, prob))
    return fold_scores


def _cv_scores_shuffled_y(name, builder, Xf, y, groups, gkf, seed=42):
    rng = np.random.default_rng(seed)
    y_shuf = pd.Series(y.values.copy(), index=y.index)
    rng.shuffle(y_shuf.values)  # embaralha alvo
    return _cv_scores(name, builder, Xf, y_shuf, groups, gkf)


def run_train():
    # 1) dados
    df = load_data(DATA_PATH)

    # 2) features e alvo
    Xf = engineer(df)
    y = make_target(df)
    Xf, y = _sanitize_features_targets(Xf, y)

    # 3) grupos (por sujeito, se existir)
    if "Subject ID" in df.columns:
        groups_raw = df.loc[Xf.index, "Subject ID"]
    else:
        groups_raw = pd.Series(np.arange(len(Xf), dtype=int), index=Xf.index)
    groups = pd.Series(groups_raw).astype("category").cat.codes  # compacto/estável

    # 4) validação cruzada
    gkf = _adaptive_gkf(groups, max_splits=5)

    candidates = [
        ("log_reg", logistic_pipe),
        ("gbm", gbm_pipe),
    ]
    results = []
    best_name, best_auc = None, -np.inf
    best_builder = None

    # CV normal
    for name, builder in candidates:
        fold_scores = _cv_scores(name, builder, Xf, y, groups, gkf)
        results.append((name, fold_scores))
        auc_mean = float(np.mean([s["auc"] for s in fold_scores]))
        if auc_mean > best_auc:
            best_auc = auc_mean
            best_name = name
            best_builder = builder

    # CV com alvo embaralhado (diagnóstico de vazamento)
    shuf_results = []
    for name, builder in candidates:
        fold_scores = _cv_scores_shuffled_y(name, builder, Xf, y, groups, gkf, seed=1337)
        shuf_results.append((name, fold_scores))

    # 5) treino final no conjunto completo (modelo “de produção”)
    assert best_builder is not None, "Nenhum modelo candidato foi construído."
    with parallel_backend("threading", n_jobs=1):
        final_model = best_builder()
        final_model.fit(Xf, y)

    # 6) salvar modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(final_model, MODEL_PATH)

    # 7) relatório rápido (inclui diagnóstico embaralhado)
    def _pack(name, folds):
        return [dict(auc=float(s["auc"]), f1=float(s["f1"])) for s in folds]

    report = {
        "best_model": best_name,
        "best_auc_cv": float(best_auc),
        "cv_details": [(n, _pack(n, fs)) for (n, fs) in results],
        "cv_details_shuffled_y": [(n, _pack(n, fs)) for (n, fs) in shuf_results],
        "n_rows_train": int(Xf.shape[0]),
        "n_cols_train": int(Xf.shape[1]),
        "n_groups": int(pd.Series(groups).nunique()),
        "groups_overlap_example": 0,  # não há overlap em CV de grupos por definição
    }

    with open(REPORTS_DIR / "train_debug.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    info = run_train()
    try:
        print(json.dumps(info, indent=2))
    except Exception:
        print(info)
