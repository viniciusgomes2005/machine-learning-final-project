import numpy as np
import pandas as pd
from datetime import time, timedelta

def _circ(x):
    if pd.isna(x): return np.nan
    rad = (x.hour + x.minute/60.)/24.*2*np.pi
    return np.cos(rad), np.sin(rad)

def engineer(df):
    df = df.copy()
    # bedtime / wakeup (codificação circular)
    bt = pd.to_datetime(df["Bedtime"], errors="coerce")
    wu = pd.to_datetime(df["Wakeup time"], errors="coerce")

    df["bt_cos"], df["bt_sin"] = zip(*bt.apply(_circ))
    df["wu_cos"], df["wu_sin"] = zip(*wu.apply(_circ))

    # time in bed (ajuste virada do dia)
    dur = (wu - bt).apply(lambda d: d.total_seconds()/3600. if pd.notna(d) else np.nan)
    dur = dur.mask(dur < 0, dur + 24.)
    df["time_in_bed"] = dur

    # ratios e metricas
    df["awak_per_h"] = df["Awakenings"] / df["Sleep duration"].replace(0, np.nan)
    df["low_duration"] = (df["Sleep duration"] < 7.).astype(int)

    # encoding simple
    df = pd.get_dummies(df, columns=["Gender", "Smoking status"], drop_first=True)

    feats = [
        "bt_cos","bt_sin","wu_cos","wu_sin","time_in_bed",
        "Sleep duration",
        "REM sleep percentage","Deep sleep percentage","Light sleep percentage",
        "Awakenings","awak_per_h","low_duration",
        "Caffeine intake","Alcohol consumption","Exercise frequency",
        "Age"
    ] + [c for c in df.columns if c.startswith("Gender_") or c.startswith("Smoking status_")]

    df = df.reindex(columns=feats)
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
    df = df.loc[:, df.notna().any()]
    const = df.nunique(dropna=False) <= 1
    df = df.loc[:, ~const]
    return df
