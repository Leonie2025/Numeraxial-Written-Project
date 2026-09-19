from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import math
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_KEY_PHRASES = {
    "inflation": r"\binflation\b",
    "core inflation": r"\bcore inflation\b",
    "price stability": r"\bprice stability\b",
    "inflation expectations": r"\binflation expectations?\b",
    "labor market": r"\blabou?r market\b",
    "employment growth": r"\bemployment growth\b",
    "unemployment rate": r"\bunemployment rate\b",
    "economic activity": r"\beconomic activity\b",
    "economic growth": r"\beconomic growth\b",
    "policy rate": r"\bpolicy rate\b",
    "federal funds rate": r"\bfederal funds rate\b",
    "interest rates": r"\binterest rates?\b",
    "rate cuts": r"\brate cuts?\b",
    "rate increases": r"\brate increases?\b",
    "rate hikes": r"\brate hikes?\b",
    "restrictive policy": r"\brestrictive (?:policy|stance)\b",
    "accommodative policy": r"\baccommodative (?:policy|stance)\b",
    "quantitative tightening": r"\bquantitative tightening\b",
    "asset purchases": r"\basset purchases?\b",
    "balance sheet": r"\bbalance sheet\b",
    "financial conditions": r"\bfinancial conditions\b",
    "downside risks": r"\bdownside risks?\b",
    "upside risks": r"\bupside risks?\b",
    "data dependent": r"\bdata[- ]dependent\b",
    "sufficiently restrictive": r"\bsufficiently restrictive\b",
    "greater confidence": r"\bgreater confidence\b",
    "easing cycle": r"\beasing cycle\b",
    "tightening cycle": r"\btightening cycle\b",
}


def split_sentences(text: str) -> list[str]:
    text = str(text).replace("\n", " ").strip()
    if not text:
        return []
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def _normal_pdf(x: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-12)
    return math.exp(-0.5 * (x / sigma) ** 2) / (
        math.sqrt(2.0 * math.pi) * sigma
    )


def regime_shift_posterior(
    delta_score: float,
    stable_sigma: float,
    prior_shift_probability: float = 0.10,
    shift_scale_multiplier: float = 3.0,
) -> float:

    prior = float(np.clip(prior_shift_probability, 1e-6, 1 - 1e-6))
    sigma0 = max(float(stable_sigma), 1e-6)
    sigma1 = max(float(shift_scale_multiplier) * sigma0, sigma0 + 1e-6)

    l0 = (1.0 - prior) * _normal_pdf(float(delta_score), sigma0)
    l1 = prior * _normal_pdf(float(delta_score), sigma1)
    denom = l0 + l1
    return float(l1 / denom) if denom > 0 else prior


def robust_delta_sigma(
    historical_scores: Sequence[float],
    sigma_floor: float = 0.03,
    fallback_sigma: float = 0.08,
    window: int = 20,
) -> float:
    scores = np.asarray(list(historical_scores), dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 3:
        return max(float(fallback_sigma), float(sigma_floor))

    delta = np.diff(scores[-(window + 1):])
    if len(delta) < 2:
        return max(float(fallback_sigma), float(sigma_floor))

    med = np.median(delta)
    mad = np.median(np.abs(delta - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(delta, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = fallback_sigma

    return max(float(sigma), float(sigma_floor))


@dataclass
class AnalyzerConfig:
    neutral_threshold: float = 0.08
    batch_size: int = 32
    max_length: int = 256
    key_phrase_top_k: int = 8
    regime_prior: float = 0.10
    regime_shift_scale_multiplier: float = 3.0
    regime_sigma_floor: float = 0.03
    regime_fallback_sigma: float = 0.08
    regime_history_window: int = 20


class CentralBankCommunicationAnalyzer:

    def __init__(
        self,
        model_dir: str | Path,
        bank: str = "FED",
        rate_beta_bp_per_delta_score: Mapping[str, float] | None = None,
        config: AnalyzerConfig | None = None,
        device: str | None = None,
        key_phrase_patterns: Mapping[str, str] | None = None,
    ):
        self.model_dir = str(model_dir)
        self.bank = str(bank).upper()
        self.config = config or AnalyzerConfig()
        self.rate_beta = (
            {str(k).lower(): float(v) for k, v in rate_beta_bp_per_delta_score.items()}
            if rate_beta_bp_per_delta_score
            else {}
        )
        self.key_phrase_patterns = dict(
            key_phrase_patterns or DEFAULT_KEY_PHRASES
        )

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )
        self.model.to(self.device)
        self.model.eval()

        # Resolve label ids from the saved model config when possible.
        label_map = {}
        for k, v in (self.model.config.id2label or {}).items():
            label_map[str(v).upper()] = int(k)
        self.id_dovish = label_map.get("DOVISH", 0)
        self.id_hawkish = label_map.get("HAWKISH", 1)
        self.id_neutral = label_map.get("NEUTRAL", 2)

    def _infer_probs(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, 3), dtype=float)

        out = []
        bs = self.config.batch_size
        for i in range(0, len(sentences), bs):
            batch = list(sentences[i:i + bs])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(
                    self.model(**enc).logits, dim=1
                ).cpu().numpy()
            out.append(p)
        return np.vstack(out)

    def _extract_key_phrases(
        self,
        sentences: Sequence[str],
        sentence_scores: np.ndarray,
    ) -> list[dict]:
        found: Dict[str, dict] = {}

        for sent, score in zip(sentences, sentence_scores):
            for phrase, pattern in self.key_phrase_patterns.items():
                if not re.search(pattern, sent, flags=re.IGNORECASE):
                    continue
                item = found.setdefault(
                    phrase,
                    {
                        "phrase": phrase,
                        "mentions": 0,
                        "scores": [],
                        "examples": [],
                    },
                )
                item["mentions"] += 1
                item["scores"].append(float(score))
                if len(item["examples"]) < 2:
                    item["examples"].append(sent[:300])

        rows = []
        for item in found.values():
            avg_score = float(np.mean(item["scores"]))
            strength = float(np.mean(np.abs(item["scores"])))
            if avg_score > 0.02:
                stance = "HAWKISH"
            elif avg_score < -0.02:
                stance = "DOVISH"
            else:
                stance = "NEUTRAL"

            rows.append(
                {
                    "phrase": item["phrase"],
                    "mentions": int(item["mentions"]),
                    "stance": stance,
                    "average_sentence_hawk_dove_score": avg_score,
                    "directional_strength": strength,
                    "example": item["examples"][0],
                }
            )

        rows.sort(
            key=lambda x: (
                x["mentions"] * (0.5 + x["directional_strength"])
            ),
            reverse=True,
        )
        return rows[: self.config.key_phrase_top_k]

    def analyze(
        self,
        text: str,
        document_date: str | None = None,
        document_type: str | None = None,
        document_id: str | None = None,
        historical_scores: Sequence[float] | None = None,
    ) -> dict:
        historical_scores = list(historical_scores or [])
        sentences = split_sentences(text)
        probs = self._infer_probs(sentences)

        if len(probs) == 0:
            mean_p = np.array([np.nan, np.nan, np.nan])
            sentence_scores = np.array([], dtype=float)
            score = 0.0
        else:
            sentence_scores = (
                probs[:, self.id_hawkish] - probs[:, self.id_dovish]
            )
            score = float(np.mean(sentence_scores))
            mean_p = np.mean(probs, axis=0)

        th = self.config.neutral_threshold
        label = (
            "HAWKISH" if score > th
            else "DOVISH" if score < -th
            else "NEUTRAL"
        )

        stance_strength = float(min(1.0, abs(score)))
        confidence = (
            float(np.nanmax(mean_p)) if len(probs) else None
        )

        previous_score = (
            float(historical_scores[-1])
            if len(historical_scores) > 0
            else None
        )
        delta_score = (
            float(score - previous_score)
            if previous_score is not None
            else None
        )

        if delta_score is None:
            regime_probability = None
            stable_sigma = None
        else:
            stable_sigma = robust_delta_sigma(
                historical_scores,
                sigma_floor=self.config.regime_sigma_floor,
                fallback_sigma=self.config.regime_fallback_sigma,
                window=self.config.regime_history_window,
            )
            regime_probability = regime_shift_posterior(
                delta_score,
                stable_sigma,
                prior_shift_probability=self.config.regime_prior,
                shift_scale_multiplier=(
                    self.config.regime_shift_scale_multiplier
                ),
            )

        rate_path = {}
        for tenor in ("1y", "5y", "10y"):
            beta = self.rate_beta.get(tenor)
            rate_path[tenor] = (
                float(beta * delta_score)
                if beta is not None and delta_score is not None
                else None
            )

        key_phrases = self._extract_key_phrases(
            sentences,
            sentence_scores,
        )

        return {
            "schema_version": "1.0",
            "bank": self.bank,
            "document_id": document_id,
            "document_date": document_date,
            "document_type": document_type,
            "label": label,
            "hawk_dove_score": score,
            "stance_strength": stance_strength,
            "model_confidence_uncalibrated": confidence,
            "class_probability_mean": {
                "dovish": (
                    float(mean_p[self.id_dovish])
                    if len(probs) else None
                ),
                "hawkish": (
                    float(mean_p[self.id_hawkish])
                    if len(probs) else None
                ),
                "neutral": (
                    float(mean_p[self.id_neutral])
                    if len(probs) else None
                ),
            },
            "n_sentences": int(len(sentences)),
            "key_phrases": key_phrases,
            "stance_update": {
                "previous_hawk_dove_score": previous_score,
                "delta_hawk_dove_score": delta_score,
            },
            "rate_path_delta_bp": rate_path,
            "regime_shift_probability": regime_probability,
            "regime_shift_model": {
                "type": "two_variance_gaussian_bayes",
                "prior_shift_probability": self.config.regime_prior,
                "stable_delta_sigma": stable_sigma,
                "shift_scale_multiplier": (
                    self.config.regime_shift_scale_multiplier
                ),
                "note": (
                    "Posterior probability under the stated two-regime model; "
                    "not externally calibrated to historical regime labels."
                ),
            },
            "model_metadata": {
                "model_dir": self.model_dir,
                "training_domain": "FOMC-labelled sentences unless a bank-specific model is supplied",
                "device": str(self.device),
            },
        }


class MultiCentralBankAnalyzer:

    def __init__(
        self,
        model_dirs: Mapping[str, str | Path],
        rate_betas: Mapping[str, Mapping[str, float]] | None = None,
        config: AnalyzerConfig | None = None,
        device: str | None = None,
    ):
        self.model_dirs = {
            str(bank).upper(): str(path)
            for bank, path in model_dirs.items()
        }
        self.rate_betas = {
            str(bank).upper(): mapping
            for bank, mapping in (rate_betas or {}).items()
        }
        self.config = config or AnalyzerConfig()
        self.device = device
        self._cache: Dict[str, CentralBankCommunicationAnalyzer] = {}

    def get(self, bank: str) -> CentralBankCommunicationAnalyzer:
        bank = str(bank).upper()
        if bank not in self.model_dirs:
            raise KeyError(
                f"No model directory configured for {bank}. "
                "Supply a separately validated model for that central bank."
            )
        if bank not in self._cache:
            self._cache[bank] = CentralBankCommunicationAnalyzer(
                model_dir=self.model_dirs[bank],
                bank=bank,
                rate_beta_bp_per_delta_score=self.rate_betas.get(bank),
                config=self.config,
                device=self.device,
            )
        return self._cache[bank]
