from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

TENORS = ("1y", "5y", "10y")


def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[m] - p[m])))


def corr(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return np.nan
    yy, pp = y[m], p[m]
    if yy.std() == 0 or pp.std() == 0:
        return np.nan
    return float(np.corrcoef(yy, pp)[0, 1])


def coverage(y, mean, sd, z):
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    m = np.isfinite(y) & np.isfinite(mean) & np.isfinite(sd) & (sd >= 0)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[m] - mean[m]) <= z * sd[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--truth",
        default="data_bayesian/synthetic_latent_truth.csv",
    )
    ap.add_argument(
        "--kalman",
        default="results_kalman_full/kalman_event_posteriors.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="results_kalman_full",
    )
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    k = pd.read_csv(args.kalman)

    truth_date = next(
        (c for c in ["date", "event_date", "release_date", "Release Date"] if c in truth.columns),
        None,
    )
    kalman_date = next(
        (c for c in ["event_date", "date", "release_date", "Release Date"] if c in k.columns),
        None,
    )
    if truth_date is None or kalman_date is None:
        raise ValueError("Could not identify date columns.")

    truth["join_date"] = pd.to_datetime(truth[truth_date], errors="coerce").dt.normalize()
    k["join_date"] = pd.to_datetime(k[kalman_date], errors="coerce").dt.normalize()

    df = truth.merge(k, on="join_date", how="inner", suffixes=("_truth", "_kalman"))
    if df.empty:
        raise RuntimeError("No matching dates between latent truth and Kalman output.")

    stage_templates = {
        "dynamic_prior": "pred_{tenor}_mean_bp",
        "raw_futures_obs": "futures_{tenor}_obs_bp",
        "after_futures": "after_futures_{tenor}_mean_bp",
        "raw_ofi_obs": "ofi_{tenor}_obs_bp",
        "after_ofi": "after_ofi_{tenor}_mean_bp",
        "raw_text_obs": "text_{tenor}_obs_bp",
        "final_posterior": "posterior_{tenor}_mean_bp",
    }

    rows = []

    for tenor in TENORS:
        y_col = f"true_{tenor}_bp"
        if y_col not in df.columns:
            raise ValueError(f"Truth file missing {y_col}")

        y = df[y_col]
        prior_col = stage_templates["dynamic_prior"].format(tenor=tenor)
        prior_rmse = rmse(y, df[prior_col])

        for stage, tmpl in stage_templates.items():
            c = tmpl.format(tenor=tenor)
            if c not in df.columns:
                continue

            stage_rmse = rmse(y, df[c])
            rows.append({
                "tenor": tenor,
                "stage": stage,
                "n": int(np.isfinite(pd.to_numeric(df[c], errors="coerce")).sum()),
                "rmse_bp": stage_rmse,
                "mae_bp": mae(y, df[c]),
                "corr_with_truth": corr(y, df[c]),
                "rmse_improvement_vs_dynamic_prior_pct": (
                    100.0 * (prior_rmse - stage_rmse) / prior_rmse
                    if np.isfinite(prior_rmse) and prior_rmse > 0 and np.isfinite(stage_rmse)
                    else np.nan
                ),
            })

    metrics = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "kalman_stage_metrics.csv", index=False)

    coverage_rows = []
    for tenor in TENORS:
        y = df[f"true_{tenor}_bp"]
        mean = df[f"posterior_{tenor}_mean_bp"]
        sd = df[f"posterior_{tenor}_sd_bp"]

        coverage_rows.append({
            "tenor": tenor,
            "mean_posterior_sd_bp": float(pd.to_numeric(sd, errors="coerce").mean()),
            "coverage_68pct": coverage(y, mean, sd, 1.0),
            "coverage_95pct": coverage(y, mean, sd, 1.96),
            "final_rmse_bp": rmse(y, mean),
        })

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(out_dir / "kalman_posterior_coverage.csv", index=False)

    summary = {
        "n_matched_events": int(len(df)),
        "simulation_only": True,
        "final_posterior_rmse_bp": {},
        "raw_futures_rmse_bp": {},
        "after_ofi_rmse_bp": {},
        "raw_text_rmse_bp": {},
        "text_incremental_rmse_improvement_pct": {},
        "text_incremental_effect": {},
        "coverage_95pct": {},
    }

    for tenor in TENORS:
        def metric(stage, col):
            r = metrics[(metrics["tenor"] == tenor) & (metrics["stage"] == stage)]
            return float(r.iloc[0][col]) if len(r) else np.nan

        summary["final_posterior_rmse_bp"][tenor] = metric("final_posterior", "rmse_bp")
        summary["raw_futures_rmse_bp"][tenor] = metric("raw_futures_obs", "rmse_bp")
        after_ofi_rmse = metric("after_ofi", "rmse_bp")
        final_rmse = metric("final_posterior", "rmse_bp")
        summary["after_ofi_rmse_bp"][tenor] = after_ofi_rmse
        summary["raw_text_rmse_bp"][tenor] = metric("raw_text_obs", "rmse_bp")

        if np.isfinite(after_ofi_rmse) and after_ofi_rmse > 0 and np.isfinite(final_rmse):
            text_inc = 100.0 * (after_ofi_rmse - final_rmse) / after_ofi_rmse
        else:
            text_inc = np.nan
        summary["text_incremental_rmse_improvement_pct"][tenor] = float(text_inc)
        summary["text_incremental_effect"][tenor] = (
            "improves" if text_inc > 0 else "worsens" if text_inc < 0 else "neutral"
        )

        c = coverage_df[coverage_df["tenor"] == tenor]
        summary["coverage_95pct"][tenor] = (
            float(c.iloc[0]["coverage_95pct"]) if len(c) else np.nan
        )

    (out_dir / "kalman_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Stage-by-stage metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== Final posterior coverage ===")
    print(coverage_df.to_string(index=False))

    print("\n=== Incremental effect of adding text after OFI ===")
    for tenor in TENORS:
        inc = summary["text_incremental_rmse_improvement_pct"][tenor]
        effect = summary["text_incremental_effect"][tenor]
        print(f"{tenor}: {inc:+.3f}% RMSE change ({effect})")

    print(f"\nWrote:")
    print(out_dir / "kalman_stage_metrics.csv")
    print(out_dir / "kalman_posterior_coverage.csv")
    print(out_dir / "kalman_evaluation_summary.json")


if __name__ == "__main__":
    main()
