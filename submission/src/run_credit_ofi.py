from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INDICES = ("CDXIG", "CDXHY")
HORIZONS = (15, 30, 45, 60)
BAR_MINUTES = 5


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    if np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def sign_accuracy(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p) & (y != 0)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y[m]) == np.sign(p[m])))


def majority_sign_accuracy(train_y, test_y):
    train_y = np.asarray(train_y, dtype=float)
    test_y = np.asarray(test_y, dtype=float)

    train_y = train_y[np.isfinite(train_y) & (train_y != 0)]
    test_y = test_y[np.isfinite(test_y) & (test_y != 0)]

    if len(train_y) == 0 or len(test_y) == 0:
        return np.nan

    majority = 1.0 if np.sum(train_y > 0) >= np.sum(train_y < 0) else -1.0
    return float(np.mean(np.sign(test_y) == majority))


def load_credit(path: Path):
    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = {
        "timestamp",
        "mid_spread_bps",
        "signed_ofi_mm",
        "total_notional_mm",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()

    df["ofi_5m"] = (
        df["signed_ofi_mm"] / df["total_notional_mm"]
    )
    pieces = []
    for _, day in df.groupby("date", sort=False):
        day = day.copy()
        signed_roll = (
            day["signed_ofi_mm"]
            .rolling(window=3, min_periods=3)
            .sum()
        )
        total_roll = (
            day["total_notional_mm"]
            .rolling(window=3, min_periods=3)
            .sum()
        )
        day["ofi_15m"] = signed_roll / total_roll

        for h in HORIZONS:
            shift_bars = h // BAR_MINUTES
            day[f"delta_spread_{h}m_bps"] = (
                day["mid_spread_bps"].shift(-shift_bars)
                - day["mid_spread_bps"]
            )
        pieces.append(day)

    return pd.concat(pieces, ignore_index=True)


def standardize_from_train(train, test, feature):
    train = train.copy()
    test = test.copy()

    mu = float(train[feature].mean())
    sd = float(train[feature].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0

    zcol = f"z_{feature}"
    train[zcol] = (train[feature] - mu) / sd
    test[zcol] = (test[feature] - mu) / sd

    return train, test, {
        "train_mean": mu,
        "train_sd": sd,
        "z_column": zcol,
    }


def fit_one(
    train,
    test,
    index_name,
    feature,
    horizon,
):
    target = f"delta_spread_{horizon}m_bps"

    train = train.dropna(subset=[feature, target]).copy()
    test = test.dropna(subset=[feature, target]).copy()

    train, test, norm = standardize_from_train(
        train, test, feature
    )
    zcol = norm["z_column"]

    Xtr = sm.add_constant(
        train[[zcol]].to_numpy(float),
        has_constant="add",
    )
    ytr = train[target].to_numpy(float)

    maxlags = horizon // BAR_MINUTES
    model = sm.OLS(ytr, Xtr).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )

    Xte = sm.add_constant(
        test[[zcol]].to_numpy(float),
        has_constant="add",
    )
    yte = test[target].to_numpy(float)
    pred = model.predict(Xte)

    sacc = sign_accuracy(yte, pred)
    baseline_acc = majority_sign_accuracy(ytr, yte)

    row = {
        "index": index_name,
        "feature": feature,
        "horizon_min": int(horizon),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "beta_ofi": float(model.params[1]),
        "beta_se_hac": float(model.bse[1]),
        "beta_pvalue_hac": float(model.pvalues[1]),
        "intercept": float(model.params[0]),
        "intercept_pvalue_hac": float(model.pvalues[0]),
        "hac_lags": int(maxlags),
        "train_r2": float(model.rsquared),
        "test_r2": float(r2_score(yte, pred)),
        "test_rmse_bps": float(
            math.sqrt(mean_squared_error(yte, pred))
        ),
        "test_mae_bps": float(
            mean_absolute_error(yte, pred)
        ),
        "test_corr": safe_corr(yte, pred),
        "test_sign_accuracy": sacc,
        "majority_sign_baseline_accuracy": baseline_acc,
        "sign_accuracy_improvement_vs_majority": (
            sacc - baseline_acc
        ),
        "train_feature_mean": norm["train_mean"],
        "train_feature_sd": norm["train_sd"],
    }

    pred_df = test[
        ["timestamp", "date", target, feature, zcol]
    ].copy()
    pred_df["index"] = index_name
    pred_df["feature"] = feature
    pred_df["horizon_min"] = horizon
    pred_df["prediction_bps"] = pred

    return row, pred_df


def compare_features(summary):
    rows = []
    for index_name in INDICES:
        for horizon in HORIZONS:
            sub = summary[
                (summary["index"] == index_name)
                & (summary["horizon_min"] == horizon)
            ].set_index("feature")

            if not {"ofi_5m", "ofi_15m"}.issubset(sub.index):
                continue

            r5 = sub.loc["ofi_5m"]
            r15 = sub.loc["ofi_15m"]

            winner = (
                "ofi_15m"
                if r15["test_r2"] > r5["test_r2"]
                else "ofi_5m"
            )

            rows.append(
                {
                    "index": index_name,
                    "horizon_min": horizon,
                    "ofi_5m_test_r2": float(r5["test_r2"]),
                    "ofi_15m_test_r2": float(r15["test_r2"]),
                    "r2_difference_15m_minus_5m": float(
                        r15["test_r2"] - r5["test_r2"]
                    ),
                    "ofi_5m_test_rmse_bps": float(
                        r5["test_rmse_bps"]
                    ),
                    "ofi_15m_test_rmse_bps": float(
                        r15["test_rmse_bps"]
                    ),
                    "ofi_5m_sign_accuracy": float(
                        r5["test_sign_accuracy"]
                    ),
                    "ofi_15m_sign_accuracy": float(
                        r15["test_sign_accuracy"]
                    ),
                    "best_feature_by_test_r2": winner,
                }
            )
    return pd.DataFrame(rows)


def build_decay_table(summary):
    """One row/index/horizon using the better of 5m and 15m OFI by test R²."""
    rows = []
    for index_name in INDICES:
        for horizon in HORIZONS:
            sub = summary[
                (summary["index"] == index_name)
                & (summary["horizon_min"] == horizon)
            ].copy()
            if sub.empty:
                continue
            best = sub.loc[sub["test_r2"].idxmax()]
            rows.append(
                {
                    "index": index_name,
                    "horizon_min": horizon,
                    "best_feature": best["feature"],
                    "test_r2": float(best["test_r2"]),
                    "test_rmse_bps": float(best["test_rmse_bps"]),
                    "test_corr": float(best["test_corr"]),
                    "test_sign_accuracy": float(
                        best["test_sign_accuracy"]
                    ),
                    "beta_ofi": float(best["beta_ofi"]),
                    "beta_pvalue_hac": float(
                        best["beta_pvalue_hac"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="data_credit_task5",
    )
    ap.add_argument(
        "--out-dir",
        default="results_credit_task5",
    )
    ap.add_argument(
        "--test-start",
        default="2024-01-01",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    prediction_frames = []
    diagnostics = {}

    for index_name in INDICES:
        path = data_dir / f"{index_name}_5min.csv.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Generate Task-5 data first with:\n"
                f"  python src/generate_all_datasets.py "
                f"--components credit"
            )

        df = load_credit(path)

        test_start = pd.Timestamp(args.test_start)
        train = df[df["date"] < test_start].copy()
        test = df[df["date"] >= test_start].copy()

        diagnostics[index_name] = {
            "rows": int(len(df)),
            "train_rows_raw": int(len(train)),
            "test_rows_raw": int(len(test)),
            "first_timestamp": str(df["timestamp"].min()),
            "last_timestamp": str(df["timestamp"].max()),
            "mean_bars_per_day": float(
                df.groupby("date").size().mean()
            ),
            "mean_spread_bps": float(
                df["mid_spread_bps"].mean()
            ),
            "mean_abs_ofi_5m": float(
                df["ofi_5m"].abs().mean()
            ),
        }

        for horizon in HORIZONS:
            for feature in ("ofi_5m", "ofi_15m"):
                row, preds = fit_one(
                    train,
                    test,
                    index_name,
                    feature,
                    horizon,
                )
                result_rows.append(row)
                prediction_frames.append(preds)

    summary = pd.DataFrame(result_rows)
    summary.to_csv(
        out_dir / "credit_ofi_model_summary.csv",
        index=False,
    )

    compact = summary[
        [
            "index",
            "feature",
            "horizon_min",
            "beta_ofi",
            "beta_pvalue_hac",
            "train_r2",
            "test_r2",
            "test_rmse_bps",
            "test_mae_bps",
            "test_corr",
            "test_sign_accuracy",
            "majority_sign_baseline_accuracy",
            "sign_accuracy_improvement_vs_majority",
        ]
    ].copy()
    compact.to_csv(
        out_dir / "credit_ofi_compact_summary.csv",
        index=False,
    )

    comparison = compare_features(summary)
    comparison.to_csv(
        out_dir / "credit_ofi_feature_comparison.csv",
        index=False,
    )

    decay = build_decay_table(summary)
    decay.to_csv(
        out_dir / "credit_ofi_decay_summary.csv",
        index=False,
    )

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(
        out_dir / "credit_ofi_oos_predictions.csv",
        index=False,
    )

    (out_dir / "credit_data_checks.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    task_summary = {
        "indices": list(INDICES),
        "horizons_minutes": list(HORIZONS),
        "features": ["ofi_5m", "ofi_15m"],
        "test_start": args.test_start,
        "target_definition": (
            "future mid-spread change within the same trading day"
        ),
        "inference": (
            "OLS coefficients with horizon-matched HAC/Newey-West "
            "standard errors"
        ),
        "expected_beta_sign_under_synthetic_dgp": "negative",
        "synthetic_data_caveat": (
            "Simulation validation only; not empirical credit alpha."
        ),
    }

    (out_dir / "task5_summary.json").write_text(
        json.dumps(task_summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Task 5 credit OFI OOS summary ===")
    print(compact.to_string(index=False))

    print("\n=== 5m vs trailing-15m OFI ===")
    print(comparison.to_string(index=False))

    print("\n=== Best feature by horizon (decay table) ===")
    print(decay.to_string(index=False))

    print(f"\nWrote results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
