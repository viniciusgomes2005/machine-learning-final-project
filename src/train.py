# src/train.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from joblib import dump, parallel_backend

from .config import DATA_PATH, MODEL_PATH
from .data import load_data
from .features import engineer
from .model import logistic_pipe, gbm_pipe, metrics

# reduz threads de BLAS/NumPy em ambientes Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def make_target(df: pd.DataFrame) -> pd.Series:
    # alvo: ineficiência do sono (< 0.85)
    return (df["Sleep efficiency"] < 0.85).astype(int)

def _sanitize_features_targets(Xf: pd.DataFrame, y: pd.Series):
   #  cópia pra segurança
    Xf = Xf.copy()

    # 1) padroniza nomes (sem espaços) e garante string
    Xf.columns = Xf.columns.map(lambda c: str(c).strip().replace(" ", "_"))

    # 2) força tudo a numérico: o que não for número vira NaN
    for col in Xf.columns:
        if not np.issubdtype(Xf[col].dtype, np.number):
            Xf[col] = pd.to_numeric(Xf[col], errors="coerce")

    # 3) remove colunas 100% NaN (ex.: uma coluna inexistente no CSV original)
    Xf = Xf.loc[:, Xf.notna().any(axis=0)]

    # 4) remove colunas constantes (variância zero)
    const_mask = Xf.nunique(dropna=False) <= 1
    if const_mask.any():
        Xf = Xf.loc[:, ~const_mask]

    # 5) descarta linhas com valores não finitos
    arr = Xf.to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(arr).all(axis=1)
    Xf = Xf.loc[finite_mask]
    y  = y.loc[Xf.index]

    # 6) se ainda restar NaN (ex.: parsing de horas), dropa linhas
    Xf = Xf.dropna(axis=0)
    y  = y.loc[Xf.index]

    return Xf, y


def _adaptive_gkf(groups: pd.Series, max_splits: int = 5) -> GroupKFold:
    # número de grupos únicos define o teto possível de folds
    n_groups = int(pd.Series(groups).nunique())
    n_splits = max(2, min(max_splits, n_groups))
    return GroupKFold(n_splits=n_splits)

def run_train():
    df = load_data(DATA_PATH)

    # 2) features e alvo
    Xf = engineer(df)
    y = make_target(df)
    Xf, y = _sanitize_features_targets(Xf, y)

    # 3) grupos (se houver múltiplas noites por sujeito)
    groups = df.get("Subject ID", pd.Series(np.arange(len(df), dtype=int), index=df.index))
    groups = pd.Series(groups).loc[Xf.index]   

    # 4) validação cruzada
    gkf = _adaptive_gkf(groups, max_splits=5)

    candidates = [
        ("log_reg", logistic_pipe),
        ("gbm", gbm_pipe),
    ]
    results = []
    best_name, best_auc = None, -np.inf
    best_builder = None

    for name, builder in candidates:
        fold_scores = []
        with parallel_backend("threading", n_jobs=1):
            for trn_idx, val_idx in gkf.split(Xf, y, groups):
                model = builder()
                X_tr, y_tr = Xf.iloc[trn_idx], y.iloc[trn_idx]
                X_va, y_va = Xf.iloc[val_idx], y.iloc[val_idx]

                model.fit(X_tr, y_tr)
                prob = model.predict_proba(X_va)[:, 1]
                fold_scores.append(metrics(y_va, prob))

        results.append((name, fold_scores))
        auc_mean = float(np.mean([s["auc"] for s in fold_scores]))
        if auc_mean > best_auc:
            best_auc = auc_mean
            best_name = name
            best_builder = builder

    #   5) treino final no conjunto completo
    assert best_builder is not None, "Nenhum modelo candidato foi construído."
    with parallel_backend("threading", n_jobs=1):
        final_model = best_builder()
        final_model.fit(Xf, y)

    #   6) salvar
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(final_model, MODEL_PATH)

    return {
        "best_model": best_name,
        "best_auc_cv": best_auc,
        "cv_details": results,
        "n_rows_train": int(Xf.shape[0]),
        "n_cols_train": int(Xf.shape[1]),
        "n_groups": int(pd.Series(groups).nunique()),
    }

if __name__ == "__main__":
    info = run_train()
    # print leve para depuração manual
    try:
        import json
        print(json.dumps(info, indent=2))
    except Exception:
        print(info)
