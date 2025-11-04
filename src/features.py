import numpy as np
import pandas as pd

def _circ(ts):
    if pd.isna(ts):
        return (np.nan, np.nan)
    rad = (ts.hour + ts.minute / 60.0) / 24.0 * 2 * np.pi
    return (np.cos(rad), np.sin(rad))

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # parse times (aceita "HH:MM" ou datetime completo)
    bt = pd.to_datetime(df.get("Bedtime"), errors="coerce")
    wu = pd.to_datetime(df.get("Wakeup time"), errors="coerce")

    # codificação circular (robusta a NaN)
    df["bt_cos"], df["bt_sin"] = zip(*bt.apply(_circ))
    df["wu_cos"], df["wu_sin"] = zip(*wu.apply(_circ))

    # duração na cama (ajustando virada do dia)
    dur_hours = (wu - bt).apply(lambda d: d.total_seconds() / 3600.0 if pd.notna(d) else np.nan)
    dur_hours = dur_hours.mask(dur_hours < 0, dur_hours + 24.0)
    df["time_in_bed"] = dur_hours

    # numéricos esperados (cria se não existir)
    numeric_expect = [
        "Sleep duration",
        "REM sleep percentage",
        "Deep sleep percentage",
        "Light sleep percentage",
        "Awakenings",
        "Caffeine intake",
        "Alcohol consumption",
        "Exercise frequency",
        "Age",
    ]
    for col in numeric_expect:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # razões e flags
    df["awak_per_h"] = df["Awakenings"] / df["Sleep duration"].replace(0, np.nan)
    df["low_duration"] = (df["Sleep duration"] < 7.0).astype("Int64").astype(float)

    # one-hot (se existirem)
    cat_cols = []
    if "Gender" in df.columns:
        cat_cols.append("Gender")
    if "Smoking status" in df.columns:
        cat_cols.append("Smoking status")
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # seleção final de features
    ohe_prefixes = ("Gender_", "Smoking status_")
    dyn_ohe = [c for c in df.columns if any(c.startswith(p) for p in ohe_prefixes)]

    feats = [
        "bt_cos", "bt_sin", "wu_cos", "wu_sin", "time_in_bed",
        "Sleep duration",
        "REM sleep percentage", "Deep sleep percentage", "Light sleep percentage",
        "Awakenings", "awak_per_h", "low_duration",
        "Caffeine intake", "Alcohol consumption", "Exercise frequency",
        "Age",
        *dyn_ohe,
    ]

    # garante presença (reindex pode introduzir colunas novas como NaN)
    df = df.reindex(columns=feats)

    # padroniza nomes para downstream
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

    # remove colunas 100% NaN e constantes
    df = df.loc[:, df.notna().any(axis=0)]
    const_mask = df.nunique(dropna=False) <= 1
    df = df.loc[:, ~const_mask]

    return df
