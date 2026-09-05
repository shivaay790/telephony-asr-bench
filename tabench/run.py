"""Dataset manifests and the benchmark runner."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .degrade import ChannelConfig, apply_channel, standard_conditions
from .metrics import NormalizerConfig, word_errors, ErrorCounts
from .providers import REGISTRY, MockProvider, Transcript, estimate_cost, PRICES, PRICES_LAST_CHECKED


@dataclass
class Utterance:
    """One audio file plus whatever we know about the speaker.

    The metadata fields are the point of the whole exercise: an aggregate WER
    tells you nothing you can act on, but "WER is 3x higher for speakers over
    60" is a roadmap item.
    """
    id: str
    audio_path: str
    reference: str
    accent: str = "unknown"
    age_band: str = "unknown"     # teens | twenties | ... | sixties_plus
    sex: str = "unknown"
    locale: str = "en"
    domain: str = "general"       # general | clinical | food_order | insurance
    duration_s: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Manifest:
    name: str
    utterances: list[Utterance] = field(default_factory=list)
    license: str = "unspecified"
    source: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        p = Path(path)
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        utts = [Utterance.from_dict(u) for u in d.get("utterances", [])]
        # resolve relative audio paths against the manifest's own directory
        for u in utts:
            ap = Path(u.audio_path)
            if not ap.is_absolute():
                u.audio_path = str((p.parent / ap).resolve())
        return cls(d.get("name", p.stem), utts,
                   d.get("license", "unspecified"), d.get("source", ""))

    def save(self, path: str | Path) -> None:
        payload = {"name": self.name, "license": self.license,
                   "source": self.source,
                   "utterances": [asdict(u) for u in self.utterances]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @property
    def total_audio_s(self) -> float:
        return sum(u.duration_s for u in self.utterances)


@dataclass
class Result:
    utterance_id: str
    provider: str
    condition: str
    reference: str
    hypothesis: str
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    latency_s: float
    rtf: float
    cost_usd: float
    accent: str
    age_band: str
    sex: str
    domain: str
    error: Optional[str] = None


class BudgetExceeded(RuntimeError):
    pass


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def run_benchmark(manifest: Manifest,
                  provider_keys: list[str],
                  conditions: Optional[list[ChannelConfig]] = None,
                  normalizer: NormalizerConfig = NormalizerConfig(),
                  max_cost_usd: float = 0.0,
                  dry_run: bool = False,
                  progress: bool = True,
                  provider_kwargs: Optional[dict] = None) -> tuple[list[Result], dict]:
    """Run every (utterance x condition x provider) cell.

    Refuses to start if the projected spend exceeds max_cost_usd. That guard
    is the difference between a $2 experiment and an unpleasant surprise,
    because the cell count is multiplicative and easy to misjudge.
    """
    conditions = conditions or standard_conditions()
    provider_kwargs = provider_kwargs or {}

    if not manifest.utterances:
        raise ValueError("manifest contains no utterances")

    # fill in durations we do not have yet, so the estimate is real
    for u in manifest.utterances:
        if u.duration_s <= 0:
            try:
                info = sf.info(u.audio_path)
                u.duration_s = info.frames / info.samplerate
            except Exception:
                u.duration_s = 0.0

    estimate = estimate_cost(provider_keys, manifest.total_audio_s, len(conditions))
    n_cells = len(manifest.utterances) * len(conditions) * len(provider_keys)

    plan = {
        "utterances": len(manifest.utterances),
        "conditions": [c.name for c in conditions],
        "providers": provider_keys,
        "cells": n_cells,
        "audio_seconds": round(manifest.total_audio_s, 1),
        "estimated_cost_usd": {k: round(v, 4) for k, v in estimate.items()},
        "prices_last_checked": PRICES_LAST_CHECKED,
        "normalizer": normalizer.describe(),
    }

    if dry_run:
        return [], plan

    total_estimate = estimate["TOTAL"]
    if total_estimate > max_cost_usd:
        raise BudgetExceeded(
            f"projected spend ${total_estimate:.2f} exceeds budget "
            f"${max_cost_usd:.2f}. Raise --max-cost, drop a provider, or cut "
            f"conditions. Nothing has been sent."
        )

    refs = {u.id: u.reference for u in manifest.utterances}
    providers = {}
    for key in provider_keys:
        cls = REGISTRY[key]
        providers[key] = (MockProvider(refs) if cls is MockProvider
                          else cls(**provider_kwargs.get(key, {})))

    results: list[Result] = []
    spent = 0.0
    done = 0
    t_start = time.time()

    for utt in manifest.utterances:
        try:
            audio, sr = load_audio(utt.audio_path)
        except Exception as exc:
            print(f"  ! skipping {utt.id}: {exc}")
            continue

        for cond in conditions:
            degraded, out_sr = apply_channel(audio, sr, cond)
            for key, prov in providers.items():
                if isinstance(prov, MockProvider):
                    prov.set_utterance(utt.id)
                tr: Transcript = prov.transcribe(degraded, out_sr)
                spent += tr.cost_usd
                counts: ErrorCounts = word_errors(utt.reference, tr.text, normalizer)
                results.append(Result(
                    utterance_id=utt.id, provider=key, condition=cond.name,
                    reference=utt.reference, hypothesis=tr.text,
                    wer=counts.rate, substitutions=counts.substitutions,
                    deletions=counts.deletions, insertions=counts.insertions,
                    reference_words=counts.reference_length,
                    latency_s=round(tr.latency_s, 4), rtf=round(tr.rtf, 4),
                    cost_usd=tr.cost_usd, accent=utt.accent,
                    age_band=utt.age_band, sex=utt.sex, domain=utt.domain,
                    error=tr.error,
                ))
                done += 1
                if progress and done % 10 == 0:
                    pct = 100.0 * done / max(n_cells, 1)
                    print(f"  {done}/{n_cells} cells ({pct:.0f}%)  "
                          f"spent ${spent:.3f}", flush=True)

                if spent > max_cost_usd > 0:
                    print(f"  ! budget reached at ${spent:.2f}; stopping early")
                    plan["stopped_early"] = True
                    plan["actual_cost_usd"] = round(spent, 4)
                    return results, plan

    plan["actual_cost_usd"] = round(spent, 4)
    plan["wall_seconds"] = round(time.time() - t_start, 1)
    return results, plan


def save_results(results: list[Result], plan: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan, "results": [asdict(r) for r in results]}
    path = out / "results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
