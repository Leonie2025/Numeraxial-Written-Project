from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FX = ("EURUSD", "AUDUSD", "USDCNH", "DXY")


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


def load_daily_ofi(data_dir: Path, ticker: str):
    path = data_dir / f"{ticker}_5min.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run the updated unified generator for FX "
            f"WITHOUT --skip-existing."
        )

    bars = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "signed_ofi_mm", "total_volume_mm"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    bars["date"] = bars["timestamp"].dt.normalize()
    daily = bars.groupby("date", as_index=False).agg(
        signed_ofi_mm=("signed_ofi_mm", "sum"),
        total_volume_mm=("total_volume_mm", "sum"),
        n_5min_bars=("timestamp", "size"),
    )
    daily["ofi_imbalance"] = (
        daily["signed_ofi_mm"] / daily["total_volume_mm"]
    )
    return daily


def train_standardize(train, test, columns):
    train = train.copy()
    test = test.copy()
    stats = {}
    for c in columns:
        mu = float(train[c].mean())
        sd = float(train[c].std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        train[f"z_{c}"] = (train[c] - mu) / sd
        test[f"z_{c}"] = (test[c] - mu) / sd
        stats[c] = {"train_mean": mu, "train_sd": sd}
    return train, test, stats


def fit_and_eval(train, test, features, model_name, ticker, hac_lags):
    Xtr = sm.add_constant(
        train[features].to_numpy(float), has_constant="add"
    )
    ytr = train["next_return_bps"].to_numpy(float)

    model = sm.OLS(ytr, Xtr).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    Xte = sm.add_constant(
        test[features].to_numpy(float), has_constant="add"
    )
    yte = test["next_return_bps"].to_numpy(float)
    pred = model.predict(Xte)

    sacc = sign_accuracy(yte, pred)
    sign_base = majority_sign_accuracy(ytr, yte)

    row = {
        "ticker": ticker,
        "model": model_name,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "test_r2": float(r2_score(yte, pred)),
        "test_rmse_bps": float(
            math.sqrt(mean_squared_error(yte, pred))
        ),
        "test_mae_bps": float(mean_absolute_error(yte, pred)),
        "test_corr": safe_corr(yte, pred),
        "test_sign_accuracy": sacc,
        "majority_sign_baseline_accuracy": sign_base,
        "sign_accuracy_improvement_vs_majority": sacc - sign_base,
        "train_r2": float(model.rsquared),
        "hac_lags": int(hac_lags),
    }

    names = ["intercept"] + features
    for i, name in enumerate(names):
        row[f"coef_{name}"] = float(model.params[i])
        row[f"se_hac_{name}"] = float(model.bse[i])
        row[f"pvalue_hac_{name}"] = float(model.pvalues[i])

    pred_df = test[["date", "next_return_bps"]].copy()
    pred_df[f"pred_{model_name}_bps"] = pred
    return row, pred_df


def build_incremental_summary(summary):
    rows = []
    for ticker in FX:
        sub = summary[summary["ticker"] == ticker].set_index("model")
        if not {"divergence", "ofi", "joint"}.issubset(sub.index):
            continue

        singles = sub.loc[["divergence", "ofi"]]
        best_r2 = singles["test_r2"].idxmax()
        best_rmse = singles["test_rmse_bps"].idxmin()
        best_sign = singles["test_sign_accuracy"].idxmax()
        joint = sub.loc["joint"]

        rows.append(
            {
                "ticker": ticker,
                "best_single_by_r2": best_r2,
                "best_single_r2": float(singles.loc[best_r2, "test_r2"]),
                "joint_r2": float(joint["test_r2"]),
                "joint_incremental_r2": float(
                    joint["test_r2"] - singles.loc[best_r2, "test_r2"]
                ),
                "best_single_by_rmse": best_rmse,
                "best_single_rmse_bps": float(
                    singles.loc[best_rmse, "test_rmse_bps"]
                ),
                "joint_rmse_bps": float(joint["test_rmse_bps"]),
                "joint_rmse_improvement_pct": float(
                    100.0
                    * (
                        singles.loc[best_rmse, "test_rmse_bps"]
                        - joint["test_rmse_bps"]
                    )
                    / singles.loc[best_rmse, "test_rmse_bps"]
                ),
                "best_single_by_sign_accuracy": best_sign,
                "best_single_sign_accuracy": float(
                    singles.loc[best_sign, "test_sign_accuracy"]
                ),
                "joint_sign_accuracy": float(joint["test_sign_accuracy"]),
                "joint_sign_accuracy_increment": float(
                    joint["test_sign_accuracy"]
                    - singles.loc[best_sign, "test_sign_accuracy"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_fx_task4")
    ap.add_argument("--out-dir", default="results_fx_task4_v2")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--hac-lags", type=int, default=5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stance = pd.read_csv(
        data_dir / "central_bank_stance.csv",
        parse_dates=["date"],
    )
    stance["date"] = stance["date"].dt.normalize()

    results = []
    all_predictions = []
    checks = {}

    for ticker in FX:
        fx = pd.read_csv(
            data_dir / f"{ticker}_daily.csv",
            parse_dates=["date"],
        )
        fx["date"] = fx["date"].dt.normalize()

        div_col = f"div_{ticker}"
        if div_col not in stance.columns:
            raise ValueError(f"central_bank_stance.csv missing {div_col}")

        ofi = load_daily_ofi(data_dir, ticker)

        df = (
            fx.merge(
                stance[["date", div_col]],
                on="date",
                how="inner",
            )
            .rename(columns={div_col: "divergence"})
            .merge(
                ofi[["date", "ofi_imbalance", "n_5min_bars"]],
                on="date",
                how="inner",
            )
        )

        df = df.dropna(
            subset=["next_return_bps", "divergence", "ofi_imbalance"]
        ).copy()

        test_start = pd.Timestamp(args.test_start)
        train = df[df["date"] < test_start].copy()
        test = df[df["date"] >= test_start].copy()

        if len(train) < 100 or len(test) < 50:
            raise ValueError(
                f"{ticker}: insufficient sample: "
                f"train={len(train)}, test={len(test)}"
            )

        train, test, norm = train_standardize(
            train, test, ["divergence", "ofi_imbalance"]
        )

        feature_corr = float(
            train[["z_divergence", "z_ofi_imbalance"]]
            .corr()
            .iloc[0, 1]
        )

        specs = [
            ("divergence", ["z_divergence"]),
            ("ofi", ["z_ofi_imbalance"]),
            ("joint", ["z_divergence", "z_ofi_imbalance"]),
        ]

        preds = test[
            ["date", "next_return_bps", "z_divergence", "z_ofi_imbalance"]
        ].copy()

        for model_name, features in specs:
            row, pred = fit_and_eval(
                train,
                test,
                features,
                model_name,
                ticker,
                args.hac_lags,
            )
            row["train_feature_corr_div_ofi"] = feature_corr
            results.append(row)
            preds = preds.merge(
                pred[["date", f"pred_{model_name}_bps"]],
                on="date",
                how="left",
            )

        preds["joint_raw_signal"] = (
            preds["z_divergence"] + preds["z_ofi_imbalance"]
        )
        preds["ticker"] = ticker
        preds.to_csv(
            out_dir / f"{ticker}_oos_predictions.csv",
            index=False,
        )
        all_predictions.append(preds)

        checks[ticker] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "mean_5min_bars_per_day": float(df["n_5min_bars"].mean()),
            "train_corr_divergence_ofi": feature_corr,
            "normalization": norm,
        }

    summary = pd.DataFrame(results)
    summary.to_csv(out_dir / "fx_model_summary.csv", index=False)

    compact = summary[
        [
            "ticker",
            "model",
            "test_r2",
            "test_rmse_bps",
            "test_mae_bps",
            "test_corr",
            "test_sign_accuracy",
            "majority_sign_baseline_accuracy",
            "sign_accuracy_improvement_vs_majority",
        ]
    ]
    compact.to_csv(out_dir / "fx_compact_summary.csv", index=False)

    incremental = build_incremental_summary(summary)
    incremental.to_csv(
        out_dir / "fx_joint_incremental_summary.csv",
        index=False,
    )

    pd.concat(all_predictions, ignore_index=True).to_csv(
        out_dir / "fx_oos_predictions_all.csv",
        index=False,
    )

    (out_dir / "fx_data_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )

    task_summary = {
        "test_start": args.test_start,
        "hac_lags": args.hac_lags,
        "tickers": list(FX),
        "models_per_ticker": ["divergence", "ofi", "joint"],
        "all_tickers_have_5min_ofi": True,
        "primary_comparison": (
            "joint divergence+OFI versus better single-signal model"
        ),
        "synthetic_data_caveat": (
            "Simulation validation only; not empirical FX alpha."
        ),
    }
    (out_dir / "task4_summary.json").write_text(
        json.dumps(task_summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Task 4 FX OOS summary ===")
    print(compact.to_string(index=False))

    print("\n=== Joint incremental value ===")
    print(incremental.to_string(index=False))

    print(f"\nWrote results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
