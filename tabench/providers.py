"""ASR providers behind one interface, with cost accounting attached.

Every provider reports its price per audio hour so a run can be costed
*before* it spends anything. The runner refuses to start if the estimate
exceeds the budget you passed.

Prices are declared per provider and are certain to drift. They live in
PRICES so they can be corrected in one place, and every report prints the
prices it used along with the date they were last checked.
"""

from __future__ import annotations

import io
import os
import time
import wave
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

PRICES_LAST_CHECKED = "2026-09-05"

# USD per hour of audio. Verify before quoting these to anyone.
PRICES: dict[str, float] = {
    "mock": 0.0,
    "whisper_local": 0.0,
    "deepgram_nova": 0.0043 * 60,      # per-minute list price, converted
    "assemblyai_best": 0.0062 * 60,
    "openai_whisper": 0.006 * 60,
}


@dataclass
class Transcript:
    text: str
    provider: str
    latency_s: float
    audio_s: float
    cost_usd: float
    error: Optional[str] = None
    raw: Optional[dict] = None

    @property
    def rtf(self) -> float:
        """Real-time factor. Under 1.0 means faster than real time, which is
        the bar for anything that has to run in a live call."""
        return self.latency_s / self.audio_s if self.audio_s else float("inf")


class Provider(Protocol):
    name: str
    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript: ...


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """16-bit PCM WAV in memory. Every provider below accepts this."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _cost(provider: str, audio_s: float) -> float:
    return PRICES.get(provider, 0.0) * audio_s / 3600.0


# --------------------------------------------------------------------------
# mock — no network, deterministic, used by the tests and the offline demo
# --------------------------------------------------------------------------

class MockProvider:
    """Degrades a known reference in proportion to how damaged the audio is.

    This exists so the whole pipeline — degradation, scoring, slicing,
    reporting — can be exercised in CI with no keys, no downloads and no
    spend. It is not a recogniser and its numbers mean nothing on their own.
    """

    name = "mock"

    def __init__(self, reference_lookup: dict[str, str], seed: int = 0):
        self._refs = reference_lookup
        self._rng = np.random.default_rng(seed)
        self._current_id: Optional[str] = None

    def set_utterance(self, utt_id: str) -> None:
        self._current_id = utt_id

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        t0 = time.perf_counter()
        audio_s = len(audio) / sr
        ref = self._refs.get(self._current_id or "", "")
        words = ref.split()

        # Damage proxy from two signals that move in opposite directions:
        # band-limiting removes high-frequency energy, while added noise
        # raises spectral flatness. Using only the first would rank a noisy
        # channel as *cleaner* than a quiet one, which is backwards.
        if len(audio) > 64:
            spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) + 1e-9
            hf = float(spec[len(spec) // 2:].sum() / spec.sum())
            flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        else:
            hf, flatness = 0.25, 0.0
        damage = float(np.clip(0.30 - hf * 1.2 + flatness * 1.8, 0.0, 0.6))

        out = []
        for w in words:
            r = self._rng.random()
            if r < damage * 0.5:
                continue                      # deletion
            elif r < damage * 0.8:
                out.append(w[::-1])           # substitution
            else:
                out.append(w)
        latency = time.perf_counter() - t0
        return Transcript(" ".join(out), self.name, latency, audio_s,
                          _cost(self.name, audio_s))


# --------------------------------------------------------------------------
# local whisper — zero marginal cost, the honest baseline
# --------------------------------------------------------------------------

class WhisperLocalProvider:
    """faster-whisper if present, else openai-whisper. Costs nothing per call,
    which makes it the right thing to sweep parameters against before you
    spend anything on a hosted API."""

    name = "whisper_local"

    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        self.model_size = model_size
        self._backend = None
        self._model = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self._device,
                                       compute_type="int8")
            self._backend = "faster_whisper"
            return
        except Exception:
            pass
        try:
            import whisper
            self._model = whisper.load_model(self.model_size)
            self._backend = "openai_whisper"
        except Exception as exc:
            raise RuntimeError(
                "No local Whisper backend available. Install one:\n"
                "  pip install faster-whisper      (recommended, CPU-friendly)\n"
                "  pip install openai-whisper\n"
                f"underlying import error: {exc}"
            ) from exc

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        self._load()
        audio_s = len(audio) / sr
        # both backends expect 16 kHz float32 mono
        if sr != 16000:
            from .degrade import resample
            audio = resample(audio, sr, 16000)
        t0 = time.perf_counter()
        try:
            if self._backend == "faster_whisper":
                segments, _ = self._model.transcribe(audio, language="en", beam_size=1)
                text = " ".join(s.text for s in segments)
            else:
                text = self._model.transcribe(audio.astype(np.float32))["text"]
            err = None
        except Exception as exc:
            text, err = "", f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - t0
        return Transcript(text.strip(), self.name, latency, audio_s, 0.0, err)


# --------------------------------------------------------------------------
# hosted providers
# --------------------------------------------------------------------------

class _HTTPProvider:
    """Shared retry/backoff and error handling for the hosted APIs.

    Network failures are recorded on the Transcript rather than raised, so one
    flaky call does not throw away a whole run — and so the report can show
    how often a provider simply failed to answer, which is itself a result.
    """

    name = "http"
    env_var = ""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0,
                 max_retries: int = 3):
        self.api_key = api_key or os.environ.get(self.env_var, "")
        self.timeout = timeout
        self.max_retries = max_retries

    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, *args, **kwargs):
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        audio_s = len(audio) / sr
        if not self.available():
            return Transcript("", self.name, 0.0, audio_s, 0.0,
                              f"missing API key (set {self.env_var})")
        wav = to_wav_bytes(audio, sr)
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                text, raw = self._post(wav, sr)
                return Transcript(text, self.name, time.perf_counter() - t0,
                                  audio_s, _cost(self.name, audio_s), None, raw)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, 8))
        return Transcript("", self.name, 0.0, audio_s, 0.0, last_err)


class DeepgramProvider(_HTTPProvider):
    name = "deepgram_nova"
    env_var = "DEEPGRAM_API_KEY"

    def __init__(self, model: str = "nova-2-phonecall", **kw):
        super().__init__(**kw)
        self.model = model

    def _post(self, wav: bytes, sr: int):
        import requests
        r = requests.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": self.model, "smart_format": "false", "punctuate": "true"},
            headers={"Authorization": f"Token {self.api_key}",
                     "Content-Type": "audio/wav"},
            data=wav, timeout=self.timeout,
        )
        r.raise_for_status()
        j = r.json()
        text = j["results"]["channels"][0]["alternatives"][0]["transcript"]
        return text, {"model": self.model}


class AssemblyAIProvider(_HTTPProvider):
    name = "assemblyai_best"
    env_var = "ASSEMBLYAI_API_KEY"

    def _post(self, wav: bytes, sr: int):
        import requests
        h = {"authorization": self.api_key}
        up = requests.post("https://api.assemblyai.com/v2/upload",
                           headers=h, data=wav, timeout=self.timeout)
        up.raise_for_status()
        url = up.json()["upload_url"]
        job = requests.post("https://api.assemblyai.com/v2/transcript",
                            headers=h, json={"audio_url": url},
                            timeout=self.timeout)
        job.raise_for_status()
        jid = job.json()["id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            poll = requests.get(f"https://api.assemblyai.com/v2/transcript/{jid}",
                                headers=h, timeout=30)
            poll.raise_for_status()
            j = poll.json()
            if j["status"] == "completed":
                return j.get("text") or "", {"id": jid}
            if j["status"] == "error":
                raise RuntimeError(j.get("error", "assemblyai error"))
            time.sleep(1.5)
        raise TimeoutError("assemblyai transcript timed out")


class OpenAIWhisperProvider(_HTTPProvider):
    name = "openai_whisper"
    env_var = "OPENAI_API_KEY"

    def _post(self, wav: bytes, sr: int):
        import requests
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": "whisper-1", "language": "en"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("text", ""), None


REGISTRY = {
    "mock": MockProvider,
    "whisper_local": WhisperLocalProvider,
    "deepgram": DeepgramProvider,
    "assemblyai": AssemblyAIProvider,
    "openai": OpenAIWhisperProvider,
}


def estimate_cost(provider_names: list[str], total_audio_s: float,
                  n_conditions: int) -> dict[str, float]:
    """Cost of a planned run, per provider and in total.

    Called before anything executes. The multiplier people forget is
    n_conditions: seven channel conditions means seven times the spend.
    """
    out = {}
    for raw in provider_names:
        cls = REGISTRY.get(raw)
        key = getattr(cls, "name", raw)
        out[raw] = _cost(key, total_audio_s * n_conditions)
    out["TOTAL"] = sum(out.values())
    return out
