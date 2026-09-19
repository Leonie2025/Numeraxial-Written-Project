from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL2ID = {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class FOMCDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.text = df["sentence"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.text[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    pred = np.argmax(eval_pred.predictions, axis=1)
    y = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_weighted": f1_score(y, pred, average="weighted"),
        "f1_macro": f1_score(y, pred, average="macro"),
    }


def label_counts(df):
    counts = df["label"].value_counts().sort_index()
    return {ID2LABEL[int(k)]: int(v) for k, v in counts.items()}


def inverse_frequency_weights(train_df):
    counts = train_df["label"].value_counts().sort_index()
    n = len(train_df)
    k = len(LABEL2ID)
    weights = []
    for cls in range(k):
        c = int(counts.get(cls, 0))
        if c == 0:
            raise ValueError(f"Class {cls} is missing from the train split.")
        weights.append(n / (k * c))
    return torch.tensor(weights, dtype=torch.float32)


def reset_classification_head(model):
    if not hasattr(model, "classifier"):
        raise AttributeError(
            f"Expected a classifier head on {model.__class__.__name__}; "
            "inspect the model architecture before proceeding."
        )

    classifier = model.classifier
    reset_count = 0

    def _reset(m):
        nonlocal reset_count
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
            reset_count += 1

    classifier.apply(_reset)
    if reset_count == 0:
        raise RuntimeError("Classifier head was found but no resettable module was reinitialized.")
    return reset_count


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights must be provided")
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_finbert")
    ap.add_argument("--out-dir", default="finbert_model_v3")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.20)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_train = pd.read_excel(data_dir / "fomc_train_5768.xlsx")
    test_df = pd.read_excel(data_dir / "fomc_test_5768.xlsx")

    required = {"sentence", "label"}
    for name, df in [("train", full_train), ("test", test_df)]:
        if not required.issubset(df.columns):
            raise ValueError(f"{name} file must contain {required}; got {list(df.columns)}")

    full_train = full_train.dropna(subset=["sentence", "label"]).copy()
    test_df = test_df.dropna(subset=["sentence", "label"]).copy()
    full_train["label"] = full_train["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df, val_df = train_test_split(
        full_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=full_train["label"],
    )

    class_weights = inverse_frequency_weights(train_df)

    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_label_counts": label_counts(train_df),
        "val_label_counts": label_counts(val_df),
        "test_label_counts": label_counts(test_df),
        "class_weights": {
            ID2LABEL[i]: float(class_weights[i]) for i in range(3)
        },
    }
    (out_dir / "split_info.json").write_text(
        json.dumps(split_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(split_info, indent=2))

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    config = AutoConfig.from_pretrained(
        "ProsusAI/finbert",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        config=config,
    )

    reset_count = reset_classification_head(model)
    print(f"Reinitialized classification head modules: {reset_count}")

    train_ds = FOMCDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = FOMCDataset(val_df, tokenizer, max_len=args.max_len)
    test_ds = FOMCDataset(test_df, tokenizer, max_len=args.max_len)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    val_metrics = trainer.evaluate(val_ds)
    (out_dir / "validation_metrics.json").write_text(
        json.dumps(val_metrics, indent=2, default=float), encoding="utf-8"
    )

    test_output = trainer.predict(test_ds)
    y_true = test_output.label_ids
    logits = test_output.predictions
    y_pred = np.argmax(logits, axis=1)

    test_metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(3)],
        digits=4,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(
        cm,
        index=["true_DOVISH", "true_HAWKISH", "true_NEUTRAL"],
        columns=["pred_DOVISH", "pred_HAWKISH", "pred_NEUTRAL"],
    ).to_csv(out_dir / "confusion_matrix.csv")

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_df = test_df.reset_index(drop=True).copy()
    pred_df["pred_label"] = y_pred
    pred_df["pred_name"] = [ID2LABEL[int(x)] for x in y_pred]
    pred_df["p_dovish"] = probs[:, 0]
    pred_df["p_hawkish"] = probs[:, 1]
    pred_df["p_neutral"] = probs[:, 2]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    trainer.save_model(out_dir / "best_model")
    tokenizer.save_pretrained(out_dir / "best_model")

    print("\nFinal untouched-test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    main()
