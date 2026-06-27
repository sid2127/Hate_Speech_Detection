from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import re
from scipy.special import expit

# =========================================
# Load Saved Model + Vectorizer
# =========================================
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

model = joblib.load("models/logistic_regression_model.pkl")


# =========================================
# Text Cleaning Function
# =========================================
def clean_text(t):

    t = str(t)

    t = re.sub(r"http\S+|www\S+", "", t)

    t = re.sub(r"@\w+", "", t)

    t = re.sub(r"#\w+", "", t)

    t = re.sub(r"[^\w\s']", " ", t)

    t = re.sub(r"\s+", " ", t)

    return t.lower().strip()


# =========================================
# Confidence Function
# =========================================
def get_confidence(model, vec, pred):

    # For Logistic Regression / probabilistic models
    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(vec)[0]

        return float(prob[pred])

    # For Linear SVM
    elif hasattr(model, "decision_function"):

        score = model.decision_function(vec)

        score = float(score[0]) if len(score.shape) == 1 else float(score[0][pred])

        prob = expit(score)

        return prob if pred == 1 else 1 - prob

    return 0.0


# =========================================
# FastAPI App
# =========================================
app = FastAPI(
    title="Toxic Comment Detection API"
)

# =========================================
# CORS
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# Request Body
# =========================================
class TextInput(BaseModel):
    text: str


# =========================================
# Home Route
# =========================================
@app.get("/")
def home():

    return {
        "message": "Toxic Comment Detection API Running ✅"
    }


# =========================================
# Prediction Route
# =========================================
@app.post("/predict")
def predict(data: TextInput):

    # Original input
    original_text = data.text

    # Clean text
    cleaned = clean_text(original_text)

    # Vectorize
    vec = tfidf_vectorizer.transform([cleaned])

    # Predict
    pred = int(model.predict(vec)[0])

    # Confidence
    conf = get_confidence(model, vec, pred)

    conf_percent = round(conf * 100, 2)

    # =========================================
    # Final Logic
    # =========================================
    if pred == 1 and conf_percent >= 60:

        final_prediction = "Toxic 🚨"

        hide_comment = True

    else:

        final_prediction = "Clean ✅"

        hide_comment = False

    # =========================================
    # Response
    # =========================================
    return {

        "input_text": original_text,

        "cleaned_text": cleaned,

        "prediction": final_prediction,

        "confidence": conf_percent,

        "hide_comment": hide_comment
    }