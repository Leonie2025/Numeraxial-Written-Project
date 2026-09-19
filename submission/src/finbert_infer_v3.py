from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID2LABEL = {0: "DOVISH", 1: "HAWKISH", 2: "NEUTRAL"}


def split_sentences(text):
    text = str(text).replace("\n", " ").strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def infer_probs(sentences, tokenizer, model, batch_size, device):
    if not sentences:
        return np.empty((0, 3), dtype=float)

    out = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        x = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        x = {k: v.to(device) for k, v in x.items()}
        with torch.no_grad():
            p = torch.softmax(model(**x).logits, dim=1).cpu().numpy()
        out.append(p)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_finbert")
    ap.add_argument("--model-dir", default="finbert_model_v3/best_model")
    ap.add_argument("--out-dir", default="results_finbert_v3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--neutral-threshold", type=float, default=0.08)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    df = pd.read_csv(Path(args.data_dir) / "fomc_communications_2006_2025.csv")

    doc_rows = []
    sent_rows = []

    for _, r in df.iterrows():
        sents = split_sentences(r.get("Text", ""))
        probs = infer_probs(sents, tokenizer, model, args.batch_size, device)

        if len(probs) == 0:
            score = 0.0
            mean_p = np.array([np.nan, np.nan, np.nan])
        else:
            score_i = probs[:, 1] - probs[:, 0]
            score = float(np.mean(score_i))
            mean_p = probs.mean(axis=0)

        if score > args.neutral_threshold:
            label = "HAWKISH"
        elif score < -args.neutral_threshold:
            label = "DOVISH"
        else:
            label = "NEUTRAL"

        stance_strength = float(min(1.0, abs(score)))

        model_confidence_uncalibrated = (
            float(np.max(mean_p)) if len(probs) else np.nan
        )

        dr1 = 15.0 * score
        dr5 = 8.0 * score
        dr10 = 4.0 * score

        doc_rows.append(
            {
                "Date": r.get("Date"),
                "Release Date": r.get("Release Date"),
                "Type": r.get("Type"),
                "label": label,
                "stance_strength": stance_strength,
                "model_confidence_uncalibrated": model_confidence_uncalibrated,
                "p_dovish_mean": float(mean_p[0]) if len(probs) else np.nan,
                "p_hawkish_mean": float(mean_p[1]) if len(probs) else np.nan,
                "p_neutral_mean": float(mean_p[2]) if len(probs) else np.nan,
                "hawk_dove_score": score,
                "n_sentences": int(len(probs)),
                "delta_r_1y_bp": dr1,
                "delta_r_5y_bp": dr5,
                "delta_r_10y_bp": dr10,
            }
        )

        for j, (sent, p) in enumerate(zip(sents, probs)):
            sent_rows.append(
                {
                    "Date": r.get("Date"),
                    "Release Date": r.get("Release Date"),
                    "Type": r.get("Type"),
                    "sentence_id": j,
                    "sentence": sent,
                    "p_dovish": float(p[0]),
                    "p_hawkish": float(p[1]),
                    "p_neutral": float(p[2]),
                    "sentence_hawk_dove_score": float(p[1] - p[0]),
                    "sentence_label": ID2LABEL[int(np.argmax(p))],
                }
            )

    docs = pd.DataFrame(doc_rows)
    sents = pd.DataFrame(sent_rows)

    docs.to_csv(out_dir / "fomc_sentiment_v3.csv", index=False)
    sents.to_csv(out_dir / "fomc_sentence_scores_v3.csv", index=False)

    summary = {
        "n_documents": int(len(docs)),
        "label_counts": {str(k): int(v) for k, v in docs["label"].value_counts().to_dict().items()},
        "label_fraction": {str(k): float(v) for k, v in docs["label"].value_counts(normalize=True).to_dict().items()},
        "score_mean": float(docs["hawk_dove_score"].mean()),
        "score_std": float(docs["hawk_dove_score"].std()),
        "score_min": float(docs["hawk_dove_score"].min()),
        "score_max": float(docs["hawk_dove_score"].max()),
        "stance_strength_mean": float(docs["stance_strength"].mean()),
        "model_confidence_uncalibrated_mean": float(
            docs["model_confidence_uncalibrated"].mean()
        ),
        "neutral_threshold": args.neutral_threshold,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDocument label counts:")
    print(docs["label"].value_counts())
    print("\nScore summary:")
    print(docs["hawk_dove_score"].describe())


if __name__ == "__main__":
    main()
