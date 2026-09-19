# Fixed Income & FX Order Flow Alpha

## Data provenance

The submission deliberately separates real public data from synthetic market data. I trained the model on the data from 2021.1.1 to 2025.12.31. 

| Component | Data type |
|---|---|
| FOMC communications / labelled hawkish-dovish sentences | Real public data |
| 1Y / 5Y / 10Y Treasury yields used for rate-path calibration | Real public FRED data |
| Rates tick/order-flow experiment | Synthetic |
| Bayesian latent truth / futures / OFI observations | Synthetic |
| FX returns / OFI / central-bank stance processes | Synthetic |
| Credit spreads / OFI | Synthetic |

Synthetic experiments are used to validate methodology and signal integration under known data-generating processes. Bloomberg data access was not available in the development environment, so I was not able to validate the framework on live Bloomberg bond and FX data within the available time. As a result, the current results are based primarily on synthetic market data and public macro/text data, and some of the reported performance metrics may be weaker or less representative than what could be obtained from a fully calibrated real-market dataset.

## Reproduction

### 1. Generate / download base datasets

```bash
python src/generate_all_datasets.py \
  --project-root . \
  --components rates,finbert,fx,credit
```

### 2. Task 1 — Rates OFI / Kyle / Hasbrouck

```bash
python src/run_task1.py \
  --data-dir data_scenario2 \
  --out-dir results_task1
```

Main outputs:

```text
results_task1/lambda_summary.csv
results_task1/event_time_metrics.csv
results_task1/prediction_summary.csv
results_task1/event_vs_pre_summary.csv
results_task1/figures/
```

For event-time Hasbrouck results, the final summary uses the median across events because short-window VAR estimates can have extreme outliers around announcement jumps.

### 3. Task 2 — Train FinBERT V3

```bash
USE_TF=0 python src/finbert_train_v3.py \
  --data-dir data_finbert \
  --out-dir finbert_model_v3
```

### 4. Run FinBERT inference

```bash
USE_TF=0 python src/finbert_infer_v3.py \
  --data-dir data_finbert \
  --model-dir finbert_model_v3/best_model \
  --out-dir results_finbert_v3
```

`stance_strength` is the magnitude of the continuous hawk/dove score.  
`model_confidence_uncalibrated` is a diagnostic and is not a calibrated probability.

### 5. Calibrate statement stance changes to Treasury-rate moves

```bash
python src/calibrate_rate_path_v2.py \
  --sentiment results_finbert_v3/fomc_sentiment_v3.csv \
  --out-dir results_rate_calibration_v2
```

The calibration is intentionally reported as weak: daily Treasury closes do not provide strong out-of-sample evidence of a deterministic text-to-yield mapping. The text signal is therefore treated as a low-precision observation in the Bayesian model rather than a direct forecast.

### 6. Generate Task 3 Bayesian inputs

```bash
python src/generate_all_datasets.py \
  --project-root . \
  --components bayesian
```

### 7. Task 3 — Run the Kalman fusion model

```bash
python src/run_bayesian_kalman.py \
  --text results_rate_calibration_v2/fomc_statement_rate_path_updates_v2.csv \
  --calibration-summary results_rate_calibration_v2/calibration_v2_summary.json \
  --futures-prior data_bayesian/synthetic_futures_prior.csv \
  --ofi data_bayesian/synthetic_ofi_rate_signal.csv \
  --out-dir results_kalman_full
```

Evaluate:

```bash
python src/evaluate_kalman.py \
  --truth data_bayesian/synthetic_latent_truth.csv \
  --kalman results_kalman_full/kalman_event_posteriors.csv \
  --out-dir results_kalman_full
```

### 8. Task 4 — FX divergence + OFI

```bash
python src/run_fx_divergence.py \
  --data-dir data_fx_task4 \
  --out-dir results_fx_task4 \
  --test-start 2024-01-01
```

### 9. Task 5 — Credit OFI

```bash
python src/run_credit_ofi.py \
  --data-dir data_credit_task5 \
  --out-dir results_credit_task5 \
  --test-start 2024-01-01
```

### 10. Structured Fed statement analyser

Create a Statement-only source file from the full communication file:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data_finbert/fomc_communications_2006_2025.csv")
df = df[df["Type"].str.lower() == "statement"].copy()
df.to_csv("data_finbert/fomc_statements_2006_2025.csv", index=False)
print("saved", len(df), "statements")
PY
```

Run:

```bash
PYTHONPATH=. USE_TF=0 python src/run_macro_analyzer.py \
  --input data_finbert/fomc_statements_2006_2025.csv \
  --model-dir finbert_model_v3/best_model \
  --bank FED \
  --calibration-summary results_rate_calibration_v2/calibration_v2_summary.json \
  --output results_macro_analyzer/fed_statements_structured.jsonl
```

The final run contains **169 Fed Statements**. The first observation has no stance delta / rate-path delta / regime-shift posterior because there is no prior statement.

The output contract is documented in:

```text
examples/central_bank_signal.schema.json
```

## Main results

### Task 1 — Rates

Full-sample OLS and daily Hasbrouck estimates recover the planted constant structural lambda closely for all five contracts.

Event-time estimates increase around macro announcements even though structural lambda is held constant in the simulation. This is interpreted as an estimator / omitted-public-news effect rather than evidence that structural liquidity was mechanically changed in the generator.

Pre-announcement OFI has only weak predictive power. Across the five contracts, out-of-sample-style event regressions show low \(R^2\) values (roughly 0.7%–3.4%), so this result is not presented as strong alpha evidence.

### Task 2 — FinBERT

Untouched test performance of the final V3 classifier:

```text
Accuracy:       0.6197
Macro F1:       0.6080
Weighted F1:    0.6233
```

Class F1:

```text
DOVISH:   0.5603
HAWKISH:  0.5825
NEUTRAL:  0.6813
```

The Treasury rate-path calibration has economically sensible full-sample signs but weak statistical and out-of-sample explanatory power.

### Task 3 — Bayesian / Kalman fusion

Final posterior RMSE (bp):

| Tenor | Raw futures | After OFI | Final (+ text) |
|---|---:|---:|---:|
| 1Y | 1.496 | 1.134 | 1.103 |
| 5Y | 1.977 | 1.401 | 1.410 |
| 10Y | 2.296 | 1.733 | 1.721 |

The text observation has a small, tenor-dependent incremental effect:

```text
1Y:   +2.74% RMSE improvement
5Y:   -0.62% (slight worsening)
10Y:  +0.72% RMSE improvement
```

The model therefore does not claim that text improves every tenor.

### Task 4 — FX

The joint divergence + OFI model improves out-of-sample \(R^2\) and RMSE over the best single signal for all four instruments.

| Instrument | Best single R² | Joint R² | Increment |
|---|---:|---:|---:|
| EURUSD | 6.85% | 11.18% | +4.33 pp |
| AUDUSD | 2.89% | 4.68% | +1.79 pp |
| USDCNH | 12.70% | 17.06% | +4.35 pp |
| DXY | 16.83% | 19.40% | +2.57 pp |

Directional accuracy improves for three of four instruments; DXY is the exception.

### Task 5 — Credit

The 5-minute OFI feature is the strongest feature at every tested horizon, and predictability decays with horizon.

| Index | 15m R² | 30m R² | 45m R² | 60m R² |
|---|---:|---:|---:|---:|
| CDX IG | 3.05% | 1.52% | 0.93% | 0.73% |
| CDX HY | 3.20% | 1.45% | 0.77% | 0.48% |

All selected OFI coefficients are negative under the simulation's convention: positive OFI corresponds to buying credit risk and predicts spread tightening. HY has larger coefficient magnitude than IG, while explanatory power remains similar because HY is noisier.

## Package layout

```text
.
├── README.md
├── src/
│   ├── generate_all_datasets.py
│   ├── run_task1.py
│   ├── finbert_train_v3.py
│   ├── finbert_infer_v3.py
│   ├── calibrate_rate_path_v2.py
│   ├── run_bayesian_kalman.py
│   ├── evaluate_kalman.py
│   ├── run_fx_divergence.py
│   ├── run_credit_ofi.py
│   └── run_macro_analyzer.py
├── nexus/
│   └── fixedincome/
│       └── ofi_macro/
│           ├── kalman.py
│           ├── signals.py
│           └── macro_analyzer.py
├── examples/
│   └── central_bank_signal.schema.json
└── sample_outputs/
    └── fed_statements_structured.jsonl
```