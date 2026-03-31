from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# ── Load Model and Scaler ──────────────────────────
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Target Names ───────────────────────────────────
target_names = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}


# ── Input Schema ───────────────────────────────────
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ── FastAPI App ────────────────────────────────────
app = FastAPI(
    title="Iris Classification API",
    description="MLOps project - Iris model",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "Iris Classification API is running ✅"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisInput):
    # prepare input
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    # scale
    features_scaled = scaler.transform(features)

    # predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return {
        "prediction": int(prediction),
        "flower_name": target_names[prediction],
        "probability": {
            "setosa": round(probability[0], 4),
            "versicolor": round(probability[1], 4),
            "virginica": round(probability[2], 4)
        }
    }