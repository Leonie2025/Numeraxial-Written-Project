from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS1,DGS5,DGS10"
)

SERIES = {
    "DGS1": "1y",
    "DGS5": "5y",
    "DGS10": "10y",
}


def load_fred_rates(rates_csv: str | None = None) -> pd.DataFrame:
    """Load DGS1/DGS5/DGS10 from a local CSV or directly from FRED."""
    source = rates_csv if rates_csv else FRED_URL
    print(f"Loading Treasury yields from: {source}")
    df = pd.read_csv(source)

    date_candidates = ["DATE", "date", "observation_date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Could not find a date column in rates file. Columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "rate_date"}).sort_values("rate_date")

    missing = [s for s in SERIES if s not in df.columns]
    if missing:
        raise ValueError(
            f"Missing FRED series {missing}. Columns: {list(df.columns)}"
        )

    for s in SERIES:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    return df[["rate_date", *SERIES.keys()]].dropna(subset=["rate_date"])


def prepare_statement_updates(
    sentiment: pd.DataFrame,
    score_col: str,
    communication_type: str,
) -> pd.DataFrame:
    required = {"Release Date", "Type", score_col}
    missing = required - set(sentiment.columns)
    if missing:
        raise ValueError(f"Sentiment CSV missing columns: {sorted(missing)}")

    x = sentiment.copy()
    x["release_date"] = pd.to_datetime(
        x["Release Date"], errors="coerce"
    ).dt.normalize()
    x[score_col] = pd.to_numeric(x[score_col], errors="coerce")

    x = x[
        x["Type"].astype(str).str.lower() == communication_type.lower()
    ].copy()

    x = x.dropna(subset=["release_date", score_col])
    x = x.sort_values(["release_date"]).reset_index(drop=True)

    x["prev_release_date"] = x["release_date"].shift(1)
    x["prev_hawk_dove_score"] = x[score_col].shift(1)
    x["delta_hawk_dove_score"] = (
        x[score_col] - x["prev_hawk_dove_score"]
    )
    x["days_since_prev_statement"] = (
        x["release_date"] - x["prev_release_date"]
    ).dt.days

    x = x.dropna(subset=["delta_hawk_dove_score"]).reset_index(drop=True)
    return x


def attach_event_rate_moves(
    events: pd.DataFrame,
    rates: pd.DataFrame,
    max_forward_days: int = 3,
) -> pd.DataFrame:
    rates = rates.sort_values("rate_date").reset_index(drop=True)
    rows = []

    for _, ev in events.iterrows():
        release_date = pd.Timestamp(ev["release_date"]).normalize()

        future = rates[
            (rates["rate_date"] >= release_date)
            & (
                rates["rate_date"]
                <= release_date + pd.Timedelta(days=max_forward_days)
            )
        ]

        current_row = None
        for _, rr in future.iterrows():
            if rr[list(SERIES.keys())].notna().any():
                current_row = rr
                break

        if current_row is None:
            continue

        actual_date = pd.Timestamp(current_row["rate_date"])
        previous = rates[rates["rate_date"] < actual_date]

        if previous.empty:
            continue

        out = ev.to_dict()
        out["actual_rate_date"] = actual_date
        out["release_to_rate_lag_days"] = int(
            (actual_date - release_date).days
        )

        for fred_series, tenor in SERIES.items():
            cur = current_row[fred_series]
            prev_candidates = previous.dropna(subset=[fred_series])

            if pd.isna(cur) or prev_candidates.empty:
                out[f"yield_{tenor}_pct"] = np.nan
                out[f"prev_yield_{tenor}_pct"] = np.nan
                out[f"delta_y_{tenor}_bp"] = np.nan
                continue

            cur = float(cur)
            prev = float(prev_candidates.iloc[-1][fred_series])

            out[f"yield_{tenor}_pct"] = cur
            out[f"prev_yield_{tenor}_pct"] = prev
            out[f"delta_y_{tenor}_bp"] = 100.0 * (cur - prev)

        rows.append(out)

    return pd.DataFrame(rows)


def safe_corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.std(a[mask]) <= 0 or np.std(b[mask]) <= 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def sign_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))
    )


def fit_hc3(x: pd.Series, y: pd.Series):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 10 or z["x"].std(ddof=1) == 0:
        return None

    X = sm.add_constant(z["x"].to_numpy(float))
    Y = z["y"].to_numpy(float)
    return sm.OLS(Y, X).fit(cov_type="HC3")


def summarize_full_sample(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    tenor: str,
) -> Dict[str, float]:
    z = df[[x_col, y_col]].dropna()
    model = fit_hc3(z[x_col], z[y_col])

    if model is None:
        return {
            "tenor": tenor,
            "n": int(len(z)),
            "alpha_bp": np.nan,
            "beta_bp_per_delta_score": np.nan,
            "beta_se_hc3": np.nan,
            "beta_pvalue_hc3": np.nan,
            "r2": np.nan,
            "corr_score_yield_move": np.nan,
            "residual_variance_bp2": np.nan,
        }

    resid = model.resid
    return {
        "tenor": tenor,
        "n": int(len(z)),
        "alpha_bp": float(model.params[0]),
        "beta_bp_per_delta_score": float(model.params[1]),
        "beta_se_hc3": float(model.bse[1]),
        "beta_pvalue_hc3": float(model.pvalues[1]),
        "r2": float(model.rsquared),
        "corr_score_yield_move": safe_corr(z[x_col], z[y_col]),
        "residual_variance_bp2": (
            float(np.var(resid, ddof=1)) if len(resid) > 1 else np.nan
        ),
    }


def majority_sign(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y != 0)]
    if len(y) == 0:
        return 1.0
    pos = np.sum(y > 0)
    neg = np.sum(y < 0)
    return 1.0 if pos >= neg else -1.0


def chronological_oos(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    tenor: str,
    holdout_start: pd.Timestamp,
) -> Dict[str, float]:
    train = df[df["release_date"] < holdout_start][
        [x_col, y_col]
    ].dropna()

    test = df[df["release_date"] >= holdout_start][
        [x_col, y_col]
    ].dropna()

    out = {
        "tenor": tenor,
        "holdout_start": str(holdout_start.date()),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
    }

    if (
        len(train) < 10
        or len(test) < 3
        or train[x_col].std(ddof=1) == 0
    ):
        out.update({
            "alpha_train": np.nan,
            "beta_train": np.nan,
            "beta_train_se_hc3": np.nan,
            "beta_train_pvalue_hc3": np.nan,
            "model_test_r2": np.nan,
            "model_test_corr": np.nan,
            "model_test_rmse_bp": np.nan,
            "model_test_mae_bp": np.nan,
            "model_test_sign_accuracy": np.nan,
            "mean_baseline_rmse_bp": np.nan,
            "mean_baseline_mae_bp": np.nan,
            "mean_baseline_sign_accuracy": np.nan,
            "majority_sign_baseline_accuracy": np.nan,
            "zero_baseline_rmse_bp": np.nan,
            "zero_baseline_mae_bp": np.nan,
        })
        return out

    X_train = sm.add_constant(train[x_col].to_numpy(float))
    y_train = train[y_col].to_numpy(float)
    model = sm.OLS(y_train, X_train).fit(cov_type="HC3")

    X_test = sm.add_constant(test[x_col].to_numpy(float))
    y_test = test[y_col].to_numpy(float)
    model_pred = model.predict(X_test)

    train_mean = float(np.mean(y_train))
    mean_pred = np.full_like(y_test, train_mean, dtype=float)

    zero_pred = np.zeros_like(y_test, dtype=float)

    maj_sign = majority_sign(y_train)
    maj_sign_pred = np.full_like(y_test, maj_sign, dtype=float)

    out.update({
        "alpha_train": float(model.params[0]),
        "beta_train": float(model.params[1]),
        "beta_train_se_hc3": float(model.bse[1]),
        "beta_train_pvalue_hc3": float(model.pvalues[1]),
        "train_residual_variance_bp2": float(
            np.var(model.resid, ddof=1)
        ),

        "model_test_r2": float(r2_score(y_test, model_pred)),
        "model_test_corr": safe_corr(y_test, model_pred),
        "model_test_rmse_bp": float(
            math.sqrt(mean_squared_error(y_test, model_pred))
        ),
        "model_test_mae_bp": float(
            mean_absolute_error(y_test, model_pred)
        ),
        "model_test_sign_accuracy": sign_accuracy(
            y_test, model_pred
        ),

        "train_mean_yield_move_bp": train_mean,
        "mean_baseline_rmse_bp": float(
            math.sqrt(mean_squared_error(y_test, mean_pred))
        ),
        "mean_baseline_mae_bp": float(
            mean_absolute_error(y_test, mean_pred)
        ),
        "mean_baseline_sign_accuracy": sign_accuracy(
            y_test, mean_pred
        ),

        "majority_train_sign": int(maj_sign),
        "majority_sign_baseline_accuracy": sign_accuracy(
            y_test, maj_sign_pred
        ),

        "zero_baseline_rmse_bp": float(
            math.sqrt(mean_squared_error(y_test, zero_pred))
        ),
        "zero_baseline_mae_bp": float(
            mean_absolute_error(y_test, zero_pred)
        ),
    })

    if out["mean_baseline_rmse_bp"] > 0:
        out["rmse_improvement_vs_mean_pct"] = (
            100.0
            * (
                out["mean_baseline_rmse_bp"]
                - out["model_test_rmse_bp"]
            )
            / out["mean_baseline_rmse_bp"]
        )
    else:
        out["rmse_improvement_vs_mean_pct"] = np.nan

    out["sign_accuracy_improvement_vs_majority"] = (
        out["model_test_sign_accuracy"]
        - out["majority_sign_baseline_accuracy"]
    )

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sentiment",
        default="results_finbert_v3/fomc_sentiment_v3.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="results_rate_calibration_v2",
    )
    ap.add_argument(
        "--rates-csv",
        default=None,
        help=(
            "Optional local DATE,DGS1,DGS5,DGS10 CSV. "
            "If omitted, download from FRED."
        ),
    )
    ap.add_argument(
        "--score-col",
        default="hawk_dove_score",
    )
    ap.add_argument(
        "--type",
        default="Statement",
        help="Communication type used for calibration.",
    )
    ap.add_argument(
        "--holdout-start",
        default="2021-01-01",
    )
    ap.add_argument(
        "--max-forward-days",
        type=int,
        default=3,
    )

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentiment = pd.read_csv(args.sentiment)

    updates = prepare_statement_updates(
        sentiment,
        score_col=args.score_col,
        communication_type=args.type,
    )

    print(
        f"Using type={args.type}; "
        f"statement updates after first-difference={len(updates)}"
    )

    rates = load_fred_rates(args.rates_csv)

    event_moves = attach_event_rate_moves(
        updates,
        rates,
        max_forward_days=args.max_forward_days,
    )

    if event_moves.empty:
        raise RuntimeError(
            "No statement updates could be matched to Treasury yields."
        )

    event_moves.to_csv(
        out_dir / "event_level_statement_updates.csv",
        index=False,
    )

    x_col = "delta_hawk_dove_score"
    holdout_start = pd.Timestamp(args.holdout_start)

    full_rows = []
    oos_rows = []

    for tenor in ["1y", "5y", "10y"]:
        y_col = f"delta_y_{tenor}_bp"

        full_rows.append(
            summarize_full_sample(
                event_moves,
                x_col,
                y_col,
                tenor,
            )
        )

        oos_rows.append(
            chronological_oos(
                event_moves,
                x_col,
                y_col,
                tenor,
                holdout_start,
            )
        )

    full_df = pd.DataFrame(full_rows)
    oos_df = pd.DataFrame(oos_rows)

    full_df.to_csv(
        out_dir / "calibration_v2_full_sample.csv",
        index=False,
    )
    oos_df.to_csv(
        out_dir / "calibration_v2_oos.csv",
        index=False,
    )

    betas = {}
    obs_variances = {}

    for tenor in ["1y", "5y", "10y"]:
        row = oos_df[oos_df["tenor"] == tenor].iloc[0]
        betas[tenor] = (
            float(row["beta_train"])
            if pd.notna(row["beta_train"])
            else np.nan
        )
        obs_variances[tenor] = (
            float(row["train_residual_variance_bp2"])
            if pd.notna(row["train_residual_variance_bp2"])
            else np.nan
        )

    calibrated = event_moves.copy()
    for tenor in ["1y", "5y", "10y"]:
        calibrated[f"delta_r_{tenor}_bp_calibrated_v2"] = (
            calibrated[x_col] * betas[tenor]
        )

    calibrated.to_csv(
        out_dir / "fomc_statement_rate_path_updates_v2.csv",
        index=False,
    )

    summary = {
        "sentiment_file": args.sentiment,
        "communication_type": args.type,
        "score_level_column": args.score_col,
        "score_update_column": x_col,
        "n_statement_updates": int(len(updates)),
        "n_matched_rate_events": int(len(event_moves)),
        "holdout_start": str(holdout_start.date()),
        "mapping_beta_bp_per_unit_delta_score": betas,
        "train_residual_variance_bp2_for_possible_kalman_R": (
            obs_variances
        ),
        "important_limitation": (
            "Daily Treasury closes are a noisy proxy for FOMC "
            "repricing and include rate-decision, statement, press-"
            "conference, and other same-day information."
        ),
    }

    (out_dir / "calibration_v2_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Full sample: delta-score calibration ===")
    print(full_df.to_string(index=False))

    print("\n=== Chronological OOS + baselines ===")
    print(oos_df.to_string(index=False))

    print("\n=== Pre-holdout mapping used downstream ===")
    for tenor in ["1y", "5y", "10y"]:
        print(
            f"{tenor}: delta_r = "
            f"{betas[tenor]:.6f} * delta_hawk_dove_score"
        )

    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
