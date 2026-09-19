from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


RATE_SPECS = {
    "TY1":  {"start_price": 112.0, "tick_size": 0.015625, "ticks_per_day": 900},
    "ZN1":  {"start_price": 110.0, "tick_size": 0.015625, "ticks_per_day": 450},
    "RX1":  {"start_price": 136.0, "tick_size": 0.010000, "ticks_per_day": 450},
    "FF1":  {"start_price":  95.0, "tick_size": 0.005000, "ticks_per_day": 240},
    "SFR1": {"start_price":  96.0, "tick_size": 0.002500, "ticks_per_day": 900},
}

BANKS = ("FED", "ECB", "RBA", "PBOC", "BOJ")
FX_TICKERS = ("EURUSD", "AUDUSD", "USDCNH", "DXY")
TENORS = ("1y", "5y", "10y")

FINBERT_URLS = {
    "fomc_communications.csv":
        "https://raw.githubusercontent.com/vtasca/fed-statement-scraping/master/communications.csv",
    "fomc_train_5768.xlsx":
        "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/master/"
        "training_data/test-and-training/training_data/lab-manual-combine-train-5768.xlsx",
    "fomc_test_5768.xlsx":
        "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/master/"
        "training_data/test-and-training/test_data/lab-manual-combine-test-5768.xlsx",
}

COMPONENT_SEED_OFFSETS = {
    "rates": 1000,
    "bayesian": 2000,
    "fx": 3000,
    "credit": 4000,
}

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def should_write(path: Path, skip_existing: bool) -> bool:
    return not (skip_existing and path.exists())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def parse_components(s: str) -> List[str]:
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    valid = {"rates", "finbert", "bayesian", "fx", "credit"}
    bad = sorted(set(items) - valid)
    if bad:
        raise ValueError(f"Unknown components: {bad}. Valid={sorted(valid)}")
    return items


def business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def nearest_business_day(date: pd.Timestamp, bdays: pd.DatetimeIndex) -> pd.Timestamp:
    if date in bdays:
        return date
    pos = bdays.searchsorted(date)
    if pos == 0:
        return bdays[0]
    if pos >= len(bdays):
        return bdays[-1]
    before = bdays[pos - 1]
    after = bdays[pos]
    return before if abs(date - before) <= abs(after - date) else after


def first_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d


def nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    while d.weekday() != weekday:
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(days=7 * (n - 1))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Task 1

RATE_SESSION_SPECS = {
    "TY1":  dict(name="10Y UST Future", start=110.50, tick_size=0.015625,
                 daily_vol=0.22, spread_ticks=1, ticks_per_day=900, session=(7, 21)),
    "ZN1":  dict(name="10Y Note Future (alt)", start=108.75, tick_size=0.015625,
                 daily_vol=0.20, spread_ticks=1, ticks_per_day=450, session=(7, 21)),
    "RX1":  dict(name="Euro-Bund Future", start=134.20, tick_size=0.01,
                 daily_vol=0.28, spread_ticks=1, ticks_per_day=450, session=(6, 20)),
    "FF1":  dict(name="Fed Funds Future", start=95.67, tick_size=0.005,
                 daily_vol=0.02, spread_ticks=1, ticks_per_day=240, session=(12, 20)),
    "SFR1": dict(name="3M SOFR Future", start=95.85, tick_size=0.0025,
                 daily_vol=0.03, spread_ticks=1, ticks_per_day=900, session=(7, 21)),
}

PRE_EVENT_MIN = 30
POST_EVENT_MIN = 15
PRIVATE_SIGNAL_RHO = 0.25
FLOW_SIGNAL_STRENGTH = 1.35
POST_DECAY_MIN = 7.5


def last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    shift = (d.weekday() - weekday) % 7
    return d - pd.Timedelta(days=shift)


def generate_macro_events(
    start: str,
    end: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rows = []
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]

    for y in range(start_ts.year, end_ts.year + 1):
        for m in range(1, 13):
            rows.append((nth_weekday(y, m, 4, 1) + pd.Timedelta(hours=13, minutes=30), "NFP"))
            rows.append((nth_weekday(y, m, 1, 2) + pd.Timedelta(hours=13, minutes=30), "CPI"))
        for m in [1, 4, 7, 10]:
            rows.append((last_weekday(y, m, 3) + pd.Timedelta(hours=13, minutes=30), "GDP"))
        for m in fomc_months:
            rows.append((nth_weekday(y, m, 2, 3) + pd.Timedelta(hours=18), "FOMC"))

    ev = pd.DataFrame(rows, columns=["timestamp", "event"]).sort_values("timestamp")
    ev = ev[
        (ev["timestamp"] >= start_ts)
        & (ev["timestamp"] < end_ts + pd.Timedelta(days=1))
    ].reset_index(drop=True)

    ev["surprise_std"] = rng.normal(0.0, 1.0, len(ev))
    eta = rng.normal(0.0, 1.0, len(ev))
    rho = PRIVATE_SIGNAL_RHO
    ev["private_signal"] = (
        rho * ev["surprise_std"]
        + math.sqrt(1.0 - rho * rho) * eta
    )
    ev.insert(0, "event_id", [f"E{i:04d}" for i in range(1, len(ev) + 1)])
    return ev


def session_timestamps(day, start_h, end_h, n, rng):
    span_seconds = (end_h - start_h) * 3600
    offsets = np.sort(rng.uniform(0, span_seconds, n))
    base = pd.Timestamp(day) + pd.Timedelta(hours=start_h)
    return base + pd.to_timedelta(offsets, unit="s")


def rate_macro_features(ts: pd.DatetimeIndex, events: pd.DataFrame):
    n = len(ts)
    flow_signal = np.zeros(n)
    release_jump = np.zeros(n)
    event_flag = np.zeros(n)

    day = pd.Timestamp(ts[0]).normalize()
    daily_events = events[events["timestamp"].dt.normalize() == day]

    for _, ev in daily_events.iterrows():
        t0 = ev["timestamp"]
        dt = np.asarray((ts - t0).total_seconds(), dtype=float) / 60.0

        pre = (dt >= -PRE_EVENT_MIN) & (dt < 0)
        if pre.any():
            ramp = (dt[pre] + PRE_EVENT_MIN) / PRE_EVENT_MIN
            flow_signal[pre] += float(ev["private_signal"]) * ramp
            event_flag[pre] = 1.0

        post = (dt >= 0) & (dt <= POST_EVENT_MIN)
        if post.any():
            decay = np.exp(-dt[post] / POST_DECAY_MIN)
            flow_signal[post] += float(ev["surprise_std"]) * decay
            event_flag[post] = 1.0

        idx = np.flatnonzero(dt >= 0)
        if len(idx) and dt[idx[0]] <= 5.0:
            release_jump[idx[0]] += float(ev["surprise_std"])

    return flow_signal, release_jump, event_flag


def generate_rate_instrument(
    ticker: str,
    cfg: dict,
    events: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    price = float(cfg["start"])
    true_lambda = float(cfg["tick_size"]) * 0.01
    macro_to_price_direction = -1.0

    for day in trading_days:
        n = int(cfg["ticks_per_day"])
        ts = pd.DatetimeIndex(
            session_timestamps(day, *cfg["session"], n, rng)
        )
        signal, release_jump, event_flag = rate_macro_features(ts, events)

        last_size = rng.integers(1, 50, n)
        p_buy = sigmoid(
            FLOW_SIGNAL_STRENGTH * macro_to_price_direction * signal
        )
        trade_sign = np.where(rng.random(n) < p_buy, 1.0, -1.0)
        signed_volume = trade_sign * last_size

        step_vol = float(cfg["daily_vol"]) / np.sqrt(n)
        noise = rng.normal(0.0, step_vol, n)
        direct_jump = (
            macro_to_price_direction
            * float(cfg["tick_size"])
            * 4.0
            * release_jump
        )
        increments = noise + true_lambda * signed_volume + direct_jump
        mid = price + np.cumsum(increments)
        price = float(mid[-1])

        half = float(cfg["spread_ticks"]) * float(cfg["tick_size"]) / 2.0
        bid = np.round((mid - half) / cfg["tick_size"]) * cfg["tick_size"]
        ask = np.round((mid + half) / cfg["tick_size"]) * cfg["tick_size"]
        last = np.where(trade_sign > 0, ask, bid)

        part = pd.DataFrame(
            {
                "timestamp": ts,
                "ticker": ticker,
                "bid_price": bid,
                "ask_price": ask,
                "last_price": last,
                "last_size": last_size,
                "bid_size": rng.integers(1, 200, n),
                "ask_size": rng.integers(1, 200, n),
                "trade_sign_true": trade_sign,
                "signed_volume_true": signed_volume,
                "macro_signal_true": signal,
                "event_window_true": event_flag,
                "lambda_true": true_lambda,
            }
        )
        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def generate_rates(
    project_root: Path,
    start: str,
    end: str,
    seed: int,
    skip_existing: bool,
) -> Dict:
    out_dir = ensure_dir(project_root / "data_scenario2")
    truth_dir = ensure_dir(project_root / "simulation_truth")
    trading_days = business_days(start, end)

    rng_macro = np.random.default_rng(
        seed + COMPONENT_SEED_OFFSETS["rates"]
    )
    events = generate_macro_events(start, end, rng_macro)
    macro_path = out_dir / "macro_events.csv"
    if should_write(macro_path, skip_existing):
        events.to_csv(macro_path, index=False)

    meta = []
    truth_rows = []

    for i, (ticker, cfg) in enumerate(RATE_SESSION_SPECS.items()):
        path = out_dir / f"{ticker}.csv.gz"
        true_lambda = float(cfg["tick_size"]) * 0.01

        if should_write(path, skip_existing):
            print(f"[rates] generating {ticker} ...")
            rng = np.random.default_rng(
                seed + COMPONENT_SEED_OFFSETS["rates"] + 100 + i
            )
            df = generate_rate_instrument(
                ticker, cfg, events, trading_days, rng
            )
            df.to_csv(path, index=False, compression="gzip")
            total_ticks = len(df)
        else:
            print(f"[rates] skip existing {path}")
            total_ticks = len(trading_days) * int(cfg["ticks_per_day"])

        meta.append(
            {
                "ticker": ticker,
                "name": cfg["name"],
                "trading_days": len(trading_days),
                "ticks_per_day": cfg["ticks_per_day"],
                "total_ticks": total_ticks,
                "true_lambda": true_lambda,
                "file": path.name,
            }
        )
        truth_rows.append(
            {
                "ticker": ticker,
                "tick_size": cfg["tick_size"],
                "true_lambda": true_lambda,
                "ticks_per_day": cfg["ticks_per_day"],
            }
        )

    pd.DataFrame(meta).to_csv(out_dir / "metadata.csv", index=False)
    pd.DataFrame(truth_rows).to_csv(
        truth_dir / "rates_truth.csv", index=False
    )

    summary = {
        "status": "generated",
        "output_dir": str(out_dir),
        "n_business_days": len(trading_days),
        "n_macro_events": len(events),
        "tickers": list(RATE_SESSION_SPECS),
        "scenario": (
            "macro-informed order flow + public announcement jump + "
            "constant structural Kyle lambda"
        ),
        "private_signal_rho": PRIVATE_SIGNAL_RHO,
    }
    json_dump(out_dir / "generation_summary.json", summary)
    return summary

# Task 2

def download_file(url: str, path: Path, skip_existing: bool) -> None:
    if skip_existing and path.exists():
        print(f"[finbert] skip existing {path}")
        return
    print(f"[finbert] downloading {path.name}")
    urllib.request.urlretrieve(url, path)


def prepare_finbert(
    project_root: Path,
    skip_existing: bool,
) -> Dict:
    out_dir = ensure_dir(project_root / "data_finbert")

    downloaded = {}
    try:
        for filename, url in FINBERT_URLS.items():
            path = out_dir / filename
            download_file(url, path, skip_existing)
            downloaded[filename] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }

        comm_path = out_dir / "fomc_communications.csv"
        filtered_path = out_dir / "fomc_communications_2006_2025.csv"

        if should_write(filtered_path, skip_existing):
            comm = pd.read_csv(comm_path)
            date_col = "Release Date" if "Release Date" in comm.columns else "Date"
            if date_col not in comm.columns:
                raise ValueError(
                    f"FOMC communications missing Date/Release Date: {list(comm.columns)}"
                )
            dt = pd.to_datetime(comm[date_col], errors="coerce")
            mask = (
                (dt >= pd.Timestamp("2006-01-01"))
                & (dt <= pd.Timestamp("2025-12-31"))
            )
            comm.loc[mask].to_csv(filtered_path, index=False)

        return {
            "status": "downloaded",
            "output_dir": str(out_dir),
            "files": downloaded,
            "note": (
                "These are real public source files, not synthetic data."
            ),
        }

    except Exception as e:

        return {
            "status": "download_failed",
            "output_dir": str(out_dir),
            "error": repr(e),
            "note": (
                "Synthetic components can still be generated. "
                "Re-run component 'finbert' with internet access."
            ),
        }


# Task 3

def parse_text_event_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next(
        (
            c for c in [
                "release_date", "Release Date", "event_date", "date", "Date"
            ]
            if c in df.columns
        ),
        None,
    )
    if date_col is None:
        raise ValueError(f"No usable date column in {path}")

    df["event_date"] = pd.to_datetime(
        df[date_col], errors="coerce"
    ).dt.normalize()
    df = df.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)

    required = [
        "delta_r_1y_bp_calibrated_v2",
        "delta_r_5y_bp_calibrated_v2",
        "delta_r_10y_bp_calibrated_v2",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing calibrated text columns: {missing}"
        )
    return df


def generate_bayesian(
    project_root: Path,
    text_events_rel: str,
    seed: int,
    skip_existing: bool,
) -> Dict:
    out_dir = ensure_dir(project_root / "data_bayesian")
    text_path = project_root / text_events_rel

    if not text_path.exists():
        summary = {
            "status": "pending",
            "reason": (
                "Calibrated FinBERT statement file does not exist yet. "
                "Run FinBERT + calibrate_rate_path_v2.py, then rerun "
                "`generate_all_datasets.py --components bayesian`."
            ),
            "expected_text_events": str(text_path),
        }
        json_dump(out_dir / "generation_summary.json", summary)
        return summary

    df = parse_text_event_file(text_path)
    rng = np.random.default_rng(seed + COMPONENT_SEED_OFFSETS["bayesian"])

    text_component = np.column_stack(
        [
            pd.to_numeric(
                df[f"delta_r_{t}_bp_calibrated_v2"], errors="coerce"
            ).fillna(0.0).to_numpy()
            for t in TENORS
        ]
    )

    n = len(df)
    rho = 0.80
    latent_sd = np.array([2.0, 2.5, 2.5], dtype=float)
    futures_sd = np.array([1.5, 2.0, 2.5], dtype=float)
    ofi_sd = np.array([3.0, 4.0, 5.0], dtype=float)

    persistent = np.zeros((n, 3))
    truth = np.zeros((n, 3))
    state = np.zeros(3)

    for i in range(n):
        state = rho * state + rng.normal(0.0, latent_sd, size=3)
        persistent[i] = state
        truth[i] = state + text_component[i]

    futures_obs = truth + rng.normal(0.0, futures_sd, size=(n, 3))
    ofi_obs = truth + rng.normal(0.0, ofi_sd, size=(n, 3))

    dates = df["event_date"].dt.strftime("%Y-%m-%d")

    truth_path = out_dir / "synthetic_latent_truth.csv"
    fut_path = out_dir / "synthetic_futures_prior.csv"
    ofi_path = out_dir / "synthetic_ofi_rate_signal.csv"

    if should_write(truth_path, skip_existing):
        truth_df = pd.DataFrame({"date": dates})
        if "delta_hawk_dove_score" in df.columns:
            truth_df["delta_hawk_dove_score"] = pd.to_numeric(
                df["delta_hawk_dove_score"], errors="coerce"
            ).fillna(0.0)
        for j, t in enumerate(TENORS):
            truth_df[f"text_component_{t}_bp"] = text_component[:, j]
            truth_df[f"persistent_component_{t}_bp"] = persistent[:, j]
            truth_df[f"true_{t}_bp"] = truth[:, j]
        truth_df.to_csv(truth_path, index=False)

    if should_write(fut_path, skip_existing):
        fut_df = pd.DataFrame({"date": dates})
        for j, t in enumerate(TENORS):
            fut_df[f"prior_{t}_bp"] = futures_obs[:, j]
            fut_df[f"var_{t}_bp2"] = futures_sd[j] ** 2
        fut_df.to_csv(fut_path, index=False)

    if should_write(ofi_path, skip_existing):
        ofi_df = pd.DataFrame({"date": dates})
        for j, t in enumerate(TENORS):
            ofi_df[f"ofi_{t}_bp"] = ofi_obs[:, j]
            ofi_df[f"var_{t}_bp2"] = ofi_sd[j] ** 2
        ofi_df.to_csv(ofi_path, index=False)

    summary = {
        "status": "generated",
        "output_dir": str(out_dir),
        "n_events": n,
        "aligned_to": str(text_path),
        "rho": rho,
        "latent_innovation_sd_bp": dict(zip(TENORS, latent_sd.tolist())),
        "futures_observation_sd_bp": dict(zip(TENORS, futures_sd.tolist())),
        "ofi_observation_sd_bp": dict(zip(TENORS, ofi_sd.tolist())),
        "note": (
            "Latent truth/futures/OFI are synthetic; FOMC dates and text "
            "components come from the calibrated FinBERT pipeline."
        ),
    }
    json_dump(out_dir / "generation_summary.json", summary)
    return summary


# Task 4

def ar_stance(
    rng: np.random.Generator,
    n: int,
    rho: float = 0.965,
    innovation_sd: float = 0.16,
) -> np.ndarray:
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0.0, innovation_sd)
    return np.tanh(x)


def zscore_full(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def make_fx_5min(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    latent_ofi: np.ndarray,
    ticker: str,
) -> pd.DataFrame:
    rows = []
    bars_per_day = 288

    for date, latent in zip(dates, latent_ofi):
        intraday = rng.normal(loc=0.22 * latent, scale=1.0, size=bars_per_day)
        total = rng.lognormal(mean=np.log(22.0), sigma=0.45, size=bars_per_day)
        buy_share = sigmoid(0.85 * intraday)
        buy = total * buy_share
        sell = total - buy
        signed = buy - sell

        ts = pd.date_range(
            pd.Timestamp(date),
            periods=bars_per_day,
            freq="5min",
        )
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "ticker": ticker,
                    "buy_volume_mm": buy,
                    "sell_volume_mm": sell,
                    "signed_ofi_mm": signed,
                    "total_volume_mm": total,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def generate_fx(
    project_root: Path,
    start: str,
    end: str,
    seed: int,
    skip_existing: bool,
) -> Dict:
    out_dir = ensure_dir(project_root / "data_fx_task4")
    truth_dir = ensure_dir(project_root / "simulation_truth")
    rng = np.random.default_rng(seed + COMPONENT_SEED_OFFSETS["fx"])

    dates = business_days(start, end)
    n = len(dates)

    stance = pd.DataFrame({"date": dates})
    for bank in BANKS:
        stance[bank] = ar_stance(rng, n)

    global_cycle = ar_stance(rng, n, rho=0.985, innovation_sd=0.08)
    loadings = {
        "FED": 0.45,
        "ECB": 0.35,
        "RBA": 0.30,
        "PBOC": -0.10,
        "BOJ": 0.15,
    }
    for bank, loading in loadings.items():
        stance[bank] = np.tanh(stance[bank] + loading * global_cycle)

    stance["div_EURUSD"] = stance["ECB"] - stance["FED"]
    stance["div_AUDUSD"] = stance["RBA"] - stance["FED"]
    stance["div_USDCNH"] = stance["FED"] - stance["PBOC"]
    stance["foreign_DXY"] = (
        0.55 * stance["ECB"]
        + 0.25 * stance["BOJ"]
        + 0.20 * stance["PBOC"]
    )
    stance["div_DXY"] = stance["FED"] - stance["foreign_DXY"]

    stance_path = out_dir / "central_bank_stance.csv"
    if should_write(stance_path, skip_existing):
        stance.to_csv(stance_path, index=False)

    daily_ofi_obs = {}
    latent_ofi_store = {}

    for ticker in FX_TICKERS:
        div_z = zscore_full(stance[f"div_{ticker}"].to_numpy())
        idio = np.zeros(n)
        for i in range(1, n):
            idio[i] = 0.35 * idio[i - 1] + rng.normal(0.0, 1.0)

        latent = 0.30 * div_z + 0.95 * idio
        latent_ofi_store[ticker] = latent

        path = out_dir / f"{ticker}_5min.csv.gz"
        if should_write(path, skip_existing):
            bars = make_fx_5min(rng, dates, latent, ticker)
            bars.to_csv(path, index=False, compression="gzip")
        else:
            bars = pd.read_csv(path, parse_dates=["timestamp"])

        bars["date"] = pd.to_datetime(bars["timestamp"]).dt.normalize()
        g = bars.groupby("date", as_index=False).agg(
            signed_ofi_mm=("signed_ofi_mm", "sum"),
            total_volume_mm=("total_volume_mm", "sum"),
        )
        g["ofi_imbalance"] = g["signed_ofi_mm"] / g["total_volume_mm"]
        daily_ofi_obs[ticker] = (
            g.set_index("date")["ofi_imbalance"]
            .reindex(dates)
            .to_numpy()
        )

    start_prices = {
        "EURUSD": 1.0850,
        "AUDUSD": 0.6520,
        "USDCNH": 7.1450,
        "DXY": 101.35,
    }
    beta_div = {
        "EURUSD": 5.0,
        "AUDUSD": 4.5,
        "USDCNH": 4.0,
        "DXY": 4.5,
    }
    beta_ofi = {
        "EURUSD": 3.5,
        "AUDUSD": 3.0,
        "USDCNH": 2.5,
        "DXY": 2.0,
    }
    noise_sd = {
        "EURUSD": 18.0,
        "AUDUSD": 20.0,
        "USDCNH": 14.0,
        "DXY": 13.0,
    }

    truth_rows = []

    for ticker in FX_TICKERS:
        div = stance[f"div_{ticker}"].to_numpy()
        signal = beta_div[ticker] * zscore_full(div)

        if ticker in daily_ofi_obs:
            ofi_z = zscore_full(daily_ofi_obs[ticker])
            signal = signal + beta_ofi[ticker] * ofi_z

        ret_bps = np.zeros(n)
        ret_bps[0] = rng.normal(0.0, noise_sd[ticker])
        ret_bps[1:] = (
            signal[:-1] + rng.normal(0.0, noise_sd[ticker], size=n - 1)
        )

        close = np.exp(
            np.log(start_prices[ticker]) + np.cumsum(ret_bps / 10000.0)
        )
        fx = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "close": close,
                "return_bps": ret_bps,
            }
        )
        fx["next_return_bps"] = fx["return_bps"].shift(-1)

        path = out_dir / f"{ticker}_daily.csv"
        if should_write(path, skip_existing):
            fx.to_csv(path, index=False)

        truth_rows.append(
            {
                "ticker": ticker,
                "beta_div_bps_per_z": beta_div[ticker],
                "beta_ofi_bps_per_z": beta_ofi.get(ticker, 0.0),
                "return_noise_sd_bps": noise_sd[ticker],
            }
        )

    pd.DataFrame(truth_rows).to_csv(
        truth_dir / "fx_truth.csv", index=False
    )

    summary = {
        "status": "generated",
        "output_dir": str(out_dir),
        "n_business_days": n,
        "fx_tickers": list(FX_TICKERS),
        "central_banks": list(BANKS),
        "divergence_definitions": {
            "EURUSD": "ECB - FED",
            "AUDUSD": "RBA - FED",
            "USDCNH": "FED - PBOC",
            "DXY": "FED - (0.55 ECB + 0.25 BOJ + 0.20 PBOC)",
        },
        "ofi_definition": (
            "All four FX instruments have synthetic 5-minute signed OFI. "
            "For DXY this is a DXY-futures-equivalent order-flow proxy."
        ),
    }
    json_dump(out_dir / "generation_summary.json", summary)
    return summary


# Task 5

def generate_credit_index(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    index_name: str,
    start_spread_bps: float,
    ofi_beta_bps_per_z: float,
    noise_sd_bps: float,
) -> pd.DataFrame:
    bars_per_day = 78  
    rows = []
    spread = float(start_spread_bps)

    for date in dates:
        ts = pd.date_range(
            pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30),
            periods=bars_per_day,
            freq="5min",
        )

        raw_flow = rng.normal(0.0, 1.0, bars_per_day)
        for i in range(1, bars_per_day):
            raw_flow[i] = 0.35 * raw_flow[i - 1] + math.sqrt(1 - 0.35**2) * raw_flow[i]

        total_notional = rng.lognormal(
            mean=np.log(18.0 if index_name == "CDXIG" else 12.0),
            sigma=0.50,
            size=bars_per_day,
        )
        buy_share = sigmoid(0.90 * raw_flow)
        buy_notional = total_notional * buy_share
        sell_notional = total_notional - buy_notional
        signed_ofi = buy_notional - sell_notional

        ofi_sd = signed_ofi.std(ddof=1)
        z_ofi = (
            (signed_ofi - signed_ofi.mean()) / ofi_sd
            if ofi_sd > 0
            else np.zeros_like(signed_ofi)
        )

        dspread = rng.normal(0.0, noise_sd_bps, bars_per_day)
        dspread[1:] += -ofi_beta_bps_per_z * z_ofi[:-1]

        mid = spread + np.cumsum(dspread)
        mid = np.maximum(mid, 5.0)
        spread = float(mid[-1])

        half_quote = 0.20 if index_name == "CDXIG" else 0.50
        bid = mid - half_quote
        ask = mid + half_quote

        rows.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "index": index_name,
                    "bid_spread_bps": bid,
                    "ask_spread_bps": ask,
                    "mid_spread_bps": mid,
                    "buy_notional_mm": buy_notional,
                    "sell_notional_mm": sell_notional,
                    "signed_ofi_mm": signed_ofi,
                    "total_notional_mm": total_notional,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def generate_credit(
    project_root: Path,
    start: str,
    end: str,
    seed: int,
    skip_existing: bool,
) -> Dict:
    out_dir = ensure_dir(project_root / "data_credit_task5")
    truth_dir = ensure_dir(project_root / "simulation_truth")
    dates = business_days(start, end)

    specs = {
        "CDXIG": {
            "start_spread_bps": 55.0,
            "ofi_beta_bps_per_z": 0.18,
            "noise_sd_bps": 0.70,
        },
        "CDXHY": {
            "start_spread_bps": 340.0,
            "ofi_beta_bps_per_z": 0.45,
            "noise_sd_bps": 1.80,
        },
    }

    truth_rows = []

    for i, (name, spec) in enumerate(specs.items()):
        path = out_dir / f"{name}_5min.csv.gz"
        if should_write(path, skip_existing):
            print(f"[credit] generating {name} ...")
            rng = np.random.default_rng(
                seed + COMPONENT_SEED_OFFSETS["credit"] + i
            )
            df = generate_credit_index(
                rng,
                dates,
                name,
                spec["start_spread_bps"],
                spec["ofi_beta_bps_per_z"],
                spec["noise_sd_bps"],
            )
            df.to_csv(path, index=False, compression="gzip")

        truth_rows.append({"index": name, **spec})

    pd.DataFrame(truth_rows).to_csv(
        truth_dir / "credit_truth.csv", index=False
    )

    summary = {
        "status": "generated",
        "output_dir": str(out_dir),
        "n_business_days": len(dates),
        "indices": list(specs),
        "bar_frequency": "5min",
        "intended_horizons_minutes": [15, 30, 45, 60],
    }
    json_dump(out_dir / "generation_summary.json", summary)
    return summary

# Main

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project-root",
        default=".",
        help="Root of fixed_income_task project.",
    )
    ap.add_argument(
        "--components",
        default="rates,finbert,bayesian,fx,credit",
        help=(
            "Comma-separated subset of: rates,finbert,bayesian,fx,credit. "
            "Default: all."
        ),
    )
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--text-events",
        default=(
            "results_rate_calibration_v2/"
            "fomc_statement_rate_path_updates_v2.csv"
        ),
        help=(
            "Downstream calibrated FinBERT statement file used only by "
            "the Bayesian component."
        ),
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not regenerate/download files that already exist.",
    )

    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    ensure_dir(root)
    ensure_dir(root / "simulation_truth")

    components = parse_components(args.components)
    results = {}

    print(f"Project root: {root}")
    print(f"Components: {components}")
    print(f"Base seed: {args.seed}")

    if "rates" in components:
        results["rates"] = generate_rates(
            root, args.start, args.end, args.seed, args.skip_existing
        )

    if "finbert" in components:
        results["finbert"] = prepare_finbert(
            root, args.skip_existing
        )

    if "bayesian" in components:
        results["bayesian"] = generate_bayesian(
            root,
            args.text_events,
            args.seed,
            args.skip_existing,
        )

    if "fx" in components:
        results["fx"] = generate_fx(
            root, args.start, args.end, args.seed, args.skip_existing
        )

    if "credit" in components:
        results["credit"] = generate_credit(
            root, args.start, args.end, args.seed, args.skip_existing
        )

    manifest = {
        "generator": "generate_all_datasets.py",
        "base_seed": args.seed,
        "date_range": [args.start, args.end],
        "components_requested": components,
        "components": results,
        "directory_contract": {
            "Task1_rates_macro": "data_scenario2/",
            "Task2_FinBERT_sources": "data_finbert/",
            "Task3_Bayesian_inputs": "data_bayesian/",
            "Task4_FX": "data_fx_task4/",
            "Task5_credit": "data_credit_task5/",
            "simulation_truth": "simulation_truth/",
        },
        "reproducibility_note": (
            "Each synthetic component uses a deterministic component-specific "
            "seed derived from the base seed. Generating one component does "
            "not consume another component's random stream."
        ),
        "data_provenance_note": (
            "Market datasets are synthetic. FinBERT corpora are downloaded "
            "public source datasets. Bayesian synthetic inputs are generated "
            "only after the calibrated FinBERT event file exists."
        ),
    }

    manifest_path = root / "dataset_manifest.json"
    json_dump(manifest_path, manifest)

    print("\n=== Dataset preparation summary ===")
    print(json.dumps(results, indent=2, default=str))
    print(f"\nManifest: {manifest_path}")

    pending = [
        name for name, payload in results.items()
        if isinstance(payload, dict) and payload.get("status") == "pending"
    ]
    if pending:
        print(
            "\nPending downstream-dependent components: "
            + ", ".join(pending)
        )


if __name__ == "__main__":
    main()
