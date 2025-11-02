from fastapi import FastAPI
from joblib import load
from pathlib import Path
import pandas as pd
from .config import MODEL_PATH
from .features import engineer

app = FastAPI()
model = load(MODEL_PATH)

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    Xf = engineer(df)
    prob = model.predict_proba(Xf)[:,1][0]
    risk = float(prob)
    return {
        "risk": risk,
        "alert": risk >= 0.85,
        "rec": recommendations(payload, risk)
    }

def recommendations(x, risk):
    tips = []
    if x.get("Caffeine intake", 0) > 0:
        tips.append("Reduzir cafeína 6h antes de dormir.")
    if x.get("Alcohol consumption", 0) > 0:
        tips.append("Evitar álcool próximo do horário de sono.")
    if x.get("Exercise frequency", 0) == 0:
        tips.append("Inserir 20–30m de exercício diário.")
    if risk >= 0.85 and not tips:
        tips.append("Manter horário regular para deitar e levantar.")
    return tips
