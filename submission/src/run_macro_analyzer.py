from pathlib import Path
import argparse
import json

import pandas as pd

from nexus.fixedincome.ofi_macro.macro_analyzer import (
    CentralBankCommunicationAnalyzer,
)


def load_rate_beta(path):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("mapping_beta_bp_per_unit_delta_score")


def choose_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"None of {candidates} found. Columns={list(df.columns)}"
        )
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--bank", default="FED")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration-summary", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--type-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--neutral-threshold", type=float, default=0.08)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if inp.suffix.lower() in {".jsonl", ".ndjson"}:
        df = pd.read_json(inp, lines=True)
    else:
        df = pd.read_csv(inp)

    text_col = args.text_col or choose_col(
        df, ["Text", "text", "statement", "content"]
    )
    date_col = args.date_col or choose_col(
        df, ["Release Date", "Date", "date", "release_date"]
    )
    type_col = args.type_col or choose_col(
        df, ["Type", "type", "document_type"], required=False
    )
    id_col = args.id_col or choose_col(
        df, ["document_id", "id"], required=False
    )

    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)

    analyzer = CentralBankCommunicationAnalyzer(
        model_dir=args.model_dir,
        bank=args.bank,
        rate_beta_bp_per_delta_score=load_rate_beta(
            args.calibration_summary
        ),
    )
    analyzer.config.neutral_threshold = args.neutral_threshold

    history = []
    rows = []

    for i, r in df.iterrows():
        result = analyzer.analyze(
            text=r[text_col],
            document_date=(
                r["_date"].isoformat()
                if pd.notna(r["_date"]) else None
            ),
            document_type=(
                None if type_col is None else str(r[type_col])
            ),
            document_id=(
                str(r[id_col])
                if id_col is not None
                else f"{args.bank}-{i:05d}"
            ),
            historical_scores=history,
        )
        history.append(result["hawk_dove_score"])
        rows.append(result)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} structured records to {out}")


if __name__ == "__main__":
    main()
