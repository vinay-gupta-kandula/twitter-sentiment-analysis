import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_batch(input_path, output_path, model_path):
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    sentiments = []
    confidences = []

    for text in df["text"].fillna("").astype(str):
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

        sentiments.append("positive" if predicted_class == 1 else "negative")
        confidences.append(round(confidence, 4))

    result = pd.DataFrame({
        "text": df["text"],
        "predicted_sentiment": sentiments,
        "confidence": confidences
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Processed {len(result)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="model_output"
    )

    args = parser.parse_args()

    predict_batch(
        args.input,
        args.output,
        args.model
    )