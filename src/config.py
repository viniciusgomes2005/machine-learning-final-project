from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "src" / "data" / "sleep.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.joblib"
THRESH = 0.85 #   cutoff para classificar eficiência baixa
