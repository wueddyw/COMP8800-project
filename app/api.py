from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import joblib
import random
from pathlib import Path

# Creating Relative Paths to project root
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data" / "NSL-KDD"
MODEL_PATH = ROOT / "model" / "rf_ids_pipeline.joblib"
TEST_PATH = DATA_DIR / "KDDTest+.txt"       # matches your folder + filename

# Dataset schema 
FEATURE_COLS = [f"f{i}" for i in range(41)]
COLS = FEATURE_COLS + ["label", "difficulty"]

app = FastAPI(title="Random Forest IDS Prototype with API")

# Load model once 
load_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)

# Load test set once 
test_load_error = None
try:
    df_test = pd.read_csv(TEST_PATH, names=COLS)
except Exception as e:
    df_test = None
    test_load_error = str(e)

class FlowRecord(BaseModel):
    data: dict  

# Returns service health status including whether model and test data loaded successfully
@app.get("/status")
def status():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "test_loaded": df_test is not None,
        "model_path": str(MODEL_PATH),
        "test_path": str(TEST_PATH),
        "model_error": load_error,
        "test_error": test_load_error,
    }

# Accepts a single flow record (f0–f40) and returns binary attack prediction with probability
@app.post("/predict")
def predict(record: FlowRecord):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    missing = [c for c in FEATURE_COLS if c not in record.data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing[:5]} ... total {len(missing)}"
        )

    X = pd.DataFrame([[record.data[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return {
        "prediction": pred,
        "prediction_text": "attack" if pred == 1 else "normal",
        "attack_probability": proba,
    }

# Randomly samples a test row returns attack type + model prediction
@app.get("/predict/random")
def predict_random(
    attack_only: bool = Query(
        False,
        description="If true, only sample attack rows from the NSL-KDD test set."
    )
):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    if df_test is None:
        raise HTTPException(status_code=500, detail=f"Test set not loaded: {test_load_error}")

    df_pool = df_test[df_test["label"] != "normal"] if attack_only else df_test
    if df_pool.empty:
        raise HTTPException(status_code=400, detail="No rows available to sample from with current filter.")

    row = df_pool.sample(n=1, random_state=None).iloc[0]  # simpler than randrange+iloc

    true_attack_type = str(row["label"])
    payload_type = "Normal" if true_attack_type == "normal" else "Attack"

    X = pd.DataFrame([row[FEATURE_COLS].to_dict()], columns=FEATURE_COLS)

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return {
        "sample_index": int(row.name),
        "true_attack_type": true_attack_type,
        "payload_type": payload_type,
        "prediction": pred,
        "prediction_text": "Attack" if pred == 1 else "Normal",
        "attack_probability": proba,
    }
