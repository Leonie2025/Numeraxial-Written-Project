from __future__ import annotations

from pathlib import Path
import argparse
import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


TENORS = ("1y", "5y", "10y")


def as_diag(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Expected exactly three tenor values: 1Y, 5Y, 10Y.")
    if np.any(arr <= 0):
        raise ValueError(f"Variances must be strictly positive; got {arr}.")
    return np.diag(arr)


@dataclass
class GaussianRatePathFilter:
    mean: np.ndarray
    cov: np.ndarray
    process_cov: np.ndarray

    def predict(self):
        self.cov = self.cov + self.process_cov
        return self.mean.copy(), self.cov.copy()

    def update(
        self,
        observation: np.ndarray,
        obs_cov: np.ndarray,
    ):
        z = np.asarray(observation, dtype=float).reshape(3)
        R = np.asarray(obs_cov, dtype=float).reshape(3, 3)

        mask = np.isfinite(z)
        idx = np.flatnonzero(mask)

        if len(idx) == 0:
            return {
                "updated": False,
                "gain_full": np.zeros((3, 3)),
                "innovation_full": np.full(3, np.nan),
                "mean": self.mean.copy(),
                "cov": self.cov.copy(),
            }

        H = np.eye(3)[idx, :]
        z_obs = z[idx]
        R_obs = R[np.ix_(idx, idx)]

        innovation = z_obs - H @ self.mean
        S = H @ self.cov @ H.T + R_obs

        try:
            K = self.cov @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.cov @ H.T @ np.linalg.pinv(S)

        self.mean = self.mean + K @ innovation

        I = np.eye(3)
        KH = K @ H
        self.cov = (
            (I - KH) @ self.cov @ (I - KH).T
            + K @ R_obs @ K.T
        )

        gain_full = np.zeros((3, 3))
        gain_full[:, idx] = K

        innovation_full = np.full(3, np.nan)
        innovation_full[idx] = innovation

        return {
            "updated": True,
            "gain_full": gain_full,
            "innovation_full": innovation_full,
            "mean": self.mean.copy(),
            "cov": self.cov.copy(),
        }


def normalize_date_col(df: pd.DataFrame, candidates) -> pd.DataFrame:
    d = df.copy()
    col = next((c for c in candidates if c in d.columns), None)
    if col is None:
        raise ValueError(
            f"Could not find date column among {candidates}; columns={list(d.columns)}"
        )
    d["event_date"] = pd.to_datetime(d[col], errors="coerce").dt.normalize()
    return d.dropna(subset=["event_date"])


def load_text(
    path: str,
    calibration_summary: Optional[str],
    default_text_var: Sequence[float],
):
    df = pd.read_csv(path)
    df = normalize_date_col(
        df,
        ["release_date", "Release Date", "date", "Date"],
    )

    preferred = {
        "1y": "delta_r_1y_bp_calibrated_v2",
        "5y": "delta_r_5y_bp_calibrated_v2",
        "10y": "delta_r_10y_bp_calibrated_v2",
    }

    missing = [c for c in preferred.values() if c not in df.columns]
    if missing:
        raise ValueError(
            "Text input should be the output of calibrate_rate_path_v2.py. "
            f"Missing columns: {missing}"
        )

    for tenor, c in preferred.items():
        df[f"text_{tenor}_bp"] = pd.to_numeric(df[c], errors="coerce")

    text_var = np.asarray(default_text_var, dtype=float)

    if calibration_summary and Path(calibration_summary).exists():
        payload = json.loads(
            Path(calibration_summary).read_text(encoding="utf-8")
        )
        rv = payload.get(
            "train_residual_variance_bp2_for_possible_kalman_R",
            {},
        )
        candidate = np.array(
            [rv.get("1y", np.nan), rv.get("5y", np.nan), rv.get("10y", np.nan)],
            dtype=float,
        )

        valid = np.isfinite(candidate) & (candidate > 0)
        text_var[valid] = candidate[valid]

    return df.sort_values("event_date"), text_var


def load_optional_futures(
    path: Optional[str],
    default_var: Sequence[float],
):
    if not path:
        return None

    df = pd.read_csv(path)
    df = normalize_date_col(df, ["date", "Date", "event_date", "release_date"])

    for tenor in TENORS:
        required = f"prior_{tenor}_bp"
        if required not in df.columns:
            raise ValueError(f"Futures prior CSV missing: {required}")
        df[required] = pd.to_numeric(df[required], errors="coerce")

    defaults = dict(zip(TENORS, default_var))
    for tenor in TENORS:
        c = f"var_{tenor}_bp2"
        if c not in df.columns:
            df[c] = defaults[tenor]
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(defaults[tenor])

    return df.sort_values("event_date")


def load_optional_ofi(
    path: Optional[str],
    default_var: Sequence[float],
):
    if not path:
        return None

    df = pd.read_csv(path)
    df = normalize_date_col(df, ["date", "Date", "event_date", "release_date"])

    for tenor in TENORS:
        required = f"ofi_{tenor}_bp"
        if required not in df.columns:
            raise ValueError(f"OFI CSV missing: {required}")
        df[required] = pd.to_numeric(df[required], errors="coerce")

    defaults = dict(zip(TENORS, default_var))
    for tenor in TENORS:
        c = f"var_{tenor}_bp2"
        if c not in df.columns:
            df[c] = defaults[tenor]
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(defaults[tenor])

    return df.sort_values("event_date")


def exact_date_lookup(df: Optional[pd.DataFrame]):
    if df is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        out[pd.Timestamp(row["event_date"])] = row
    return out


def vec_from_row(row, prefix):
    if row is None:
        return np.full(3, np.nan)
    return np.array(
        [row.get(f"{prefix}_{t}_bp", np.nan) for t in TENORS],
        dtype=float,
    )


def var_from_row(row, default_var):
    if row is None:
        return as_diag(default_var)
    vals = []
    for t, default in zip(TENORS, default_var):
        x = row.get(f"var_{t}_bp2", default)
        x = float(x) if pd.notna(x) else float(default)
        vals.append(x if x > 0 else float(default))
    return as_diag(vals)


def add_state_columns(out, prefix, mean, cov):
    for i, tenor in enumerate(TENORS):
        out[f"{prefix}_{tenor}_mean_bp"] = float(mean[i])
        out[f"{prefix}_{tenor}_sd_bp"] = float(np.sqrt(max(cov[i, i], 0.0)))


def add_gain_columns(out, source, gain):
    for i, tenor in enumerate(TENORS):
        # Diagonal term is easiest to interpret when H = I / tenor-specific obs.
        out[f"k_{source}_{tenor}"] = float(gain[i, i])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--text",
        default="results_rate_calibration_v2/fomc_statement_rate_path_updates_v2.csv",
    )
    ap.add_argument(
        "--calibration-summary",
        default="results_rate_calibration_v2/calibration_v2_summary.json",
    )
    ap.add_argument("--futures-prior", default=None)
    ap.add_argument("--ofi", default=None)
    ap.add_argument("--out-dir", default="results_kalman")

    ap.add_argument(
        "--initial-var",
        default="25,25,25",
        help="Initial state variances in bp^2: 1Y,5Y,10Y.",
    )
    ap.add_argument(
        "--process-var",
        default="4,4,4",
        help="Random-walk process variances per FOMC event in bp^2.",
    )
    ap.add_argument(
        "--futures-var",
        default="4,6.25,9",
        help="Fallback futures observation variances in bp^2.",
    )
    ap.add_argument(
        "--ofi-var",
        default="16,25,36",
        help="Fallback OFI observation variances in bp^2.",
    )
    ap.add_argument(
        "--text-var",
        default="36,64,64",
        help="Fallback text variances in bp^2 if calibration summary is absent.",
    )
    ap.add_argument(
        "--initial-mean",
        default="0,0,0",
        help="Initial 1Y,5Y,10Y rate-path shift in bp.",
    )

    args = ap.parse_args()

    def parse3(s):
        vals = [float(x.strip()) for x in s.split(",")]
        if len(vals) != 3:
            raise ValueError("Expected 3 comma-separated values.")
        return np.array(vals, dtype=float)

    initial_mean = parse3(args.initial_mean)
    initial_var = parse3(args.initial_var)
    process_var = parse3(args.process_var)
    futures_default_var = parse3(args.futures_var)
    ofi_default_var = parse3(args.ofi_var)
    text_default_var = parse3(args.text_var)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text, text_var = load_text(
        args.text,
        args.calibration_summary,
        text_default_var,
    )
    futures = load_optional_futures(
        args.futures_prior,
        futures_default_var,
    )
    ofi = load_optional_ofi(
        args.ofi,
        ofi_default_var,
    )

    fut_lookup = exact_date_lookup(futures)
    ofi_lookup = exact_date_lookup(ofi)

    filt = GaussianRatePathFilter(
        mean=initial_mean.copy(),
        cov=as_diag(initial_var),
        process_cov=as_diag(process_var),
    )

    rows = []

    for _, tr in text.sort_values("event_date").iterrows():
        date = pd.Timestamp(tr["event_date"])
        out = {"event_date": date}
        pred_mean, pred_cov = filt.predict()
        add_state_columns(out, "pred", pred_mean, pred_cov)

        fr = fut_lookup.get(date)
        if fr is not None:
            z_fut = vec_from_row(fr, "prior")
            R_fut = var_from_row(fr, futures_default_var)

            for i, tenor in enumerate(TENORS):
                out[f"futures_{tenor}_obs_bp"] = z_fut[i]

            res = filt.update(z_fut, R_fut)
            add_gain_columns(out, "futures", res["gain_full"])
        else:
            z_fut = np.full(3, np.nan)
            for tenor in TENORS:
                out[f"futures_{tenor}_obs_bp"] = np.nan
                out[f"k_futures_{tenor}"] = 0.0

        add_state_columns(out, "after_futures", filt.mean, filt.cov)

        orow = ofi_lookup.get(date)
        if orow is not None:
            z_ofi = vec_from_row(orow, "ofi")
            R_ofi = var_from_row(orow, ofi_default_var)

            for i, tenor in enumerate(TENORS):
                out[f"ofi_{tenor}_obs_bp"] = z_ofi[i]

            res = filt.update(z_ofi, R_ofi)
            add_gain_columns(out, "ofi", res["gain_full"])
        else:
            z_ofi = np.full(3, np.nan)
            for tenor in TENORS:
                out[f"ofi_{tenor}_obs_bp"] = np.nan
                out[f"k_ofi_{tenor}"] = 0.0

        before_text_mean = filt.mean.copy()
        add_state_columns(out, "after_ofi", filt.mean, filt.cov)

        z_text = np.array(
            [
                tr["text_1y_bp"],
                tr["text_5y_bp"],
                tr["text_10y_bp"],
            ],
            dtype=float,
        )
        R_text = as_diag(text_var)

        for i, tenor in enumerate(TENORS):
            out[f"text_{tenor}_obs_bp"] = z_text[i]
            out[f"text_{tenor}_var_bp2"] = float(text_var[i])

        res_text = filt.update(z_text, R_text)
        add_gain_columns(out, "text", res_text["gain_full"])

        for i, tenor in enumerate(TENORS):
            out[f"text_contribution_{tenor}_bp"] = float(
                filt.mean[i] - before_text_mean[i]
            )

        add_state_columns(out, "posterior", filt.mean, filt.cov)

        if "delta_hawk_dove_score" in tr:
            out["delta_hawk_dove_score"] = float(
                tr["delta_hawk_dove_score"]
            )

        rows.append(out)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "kalman_event_posteriors.csv", index=False)

    gain_summary = {}
    for source in ["futures", "ofi", "text"]:
        gain_summary[source] = {}
        for tenor in TENORS:
            c = f"k_{source}_{tenor}"
            gain_summary[source][tenor] = float(
                result[c].mean()
            ) if c in result else 0.0

    posterior_sd = {}
    for tenor in TENORS:
        posterior_sd[tenor] = float(
            result[f"posterior_{tenor}_sd_bp"].mean()
        )

    summary = {
        "n_events": int(len(result)),
        "state_definition": [
            "delta_r_1y_bp",
            "delta_r_5y_bp",
            "delta_r_10y_bp",
        ],
        "futures_prior_supplied": bool(args.futures_prior),
        "ofi_supplied": bool(args.ofi),
        "text_input": args.text,
        "text_variance_bp2": {
            t: float(v) for t, v in zip(TENORS, text_var)
        },
        "process_variance_bp2": {
            t: float(v) for t, v in zip(TENORS, process_var)
        },
        "average_kalman_gain": gain_summary,
        "average_posterior_sd_bp": posterior_sd,
        "interpretation": (
            "A noisy text calibration should produce a relatively small text "
            "Kalman gain. Futures and OFI inputs are only fused when explicitly "
            "provided on matching dates; unrelated synthetic calendars are not "
            "silently joined."
        ),
    }

    (out_dir / "kalman_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Bayesian/Kalman fusion summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_dir / 'kalman_event_posteriors.csv'}")
    print(f"Wrote: {out_dir / 'kalman_summary.json'}")

    if not args.futures_prior:
        print(
            "\nNOTE: no explicit futures-prior CSV was supplied; "
            "the prior is the propagated previous posterior."
        )
    if not args.ofi:
        print(
            "NOTE: no explicit OFI CSV was supplied; "
            "no unrelated synthetic OFI series was auto-joined."
        )


if __name__ == "__main__":
    main()
