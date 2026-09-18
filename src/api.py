import os

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model_output")

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    version="1.0.0"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    sentiment = "positive" if predicted_class == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }