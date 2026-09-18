import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


MODEL_NAME = "bert-base-uncased"
TRAIN_FILE = Path("data/processed/train.csv")
TEST_FILE = Path("data/processed/test.csv")
MODEL_DIR = Path("model_output")
RESULTS_DIR = Path("results")

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    return train_dataset, test_dataset


def main():
    print("Starting BERT fine-tuning...")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset, test_dataset = load_data()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )

        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()

    final_metrics = {
        "accuracy": float(metrics["eval_accuracy"]),
        "precision": float(metrics["eval_precision"]),
        "recall": float(metrics["eval_recall"]),
        "f1_score": float(metrics["eval_f1_score"]),
    }

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    run_summary = {
        "hyperparameters": {
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": EPOCHS,
        },
        "final_metrics": {
            "accuracy": final_metrics["accuracy"],
            "f1_score": final_metrics["f1_score"],
        },
    }

    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\nTraining completed successfully.")
    print("Model saved to:", MODEL_DIR)
    print("Metrics saved to:", RESULTS_DIR / "metrics.json")
    print("Summary saved to:", RESULTS_DIR / "run_summary.json")
    print("\nFinal metrics:")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
