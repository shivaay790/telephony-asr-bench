"""Telephony channel simulation.

Public ASR benchmarks are almost always reported on clean, wideband,
close-mic audio. Production voice agents receive something quite different:
8 kHz, band-limited to roughly 300-3400 Hz, companded to 8 bits by G.711,
carried over a lossy network, and often pre-processed by noise suppression
that was tuned for human listeners rather than for a recogniser.

Each transform here is a separate, individually toggleable stage so a report
can attribute error to a specific part of the chain rather than to "the
phone".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
from scipy import signal

TELEPHONY_SR = 8000
NARROWBAND_LOW_HZ = 300.0
NARROWBAND_HIGH_HZ = 3400.0


# --------------------------------------------------------------------------
# individual stages
# --------------------------------------------------------------------------

def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling. Rational ratios only, which covers every
    sample rate pair that occurs in telephony."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = np.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def bandlimit(x: np.ndarray, sr: int,
              low_hz: float = NARROWBAND_LOW_HZ,
              high_hz: float = NARROWBAND_HIGH_HZ) -> np.ndarray:
    """Butterworth bandpass approximating the POTS passband.

    The high-pass edge matters more than people expect: it removes the
    fundamental of most adult male voices (85-155 Hz) and a good part of
    the first formant, so pitch has to be inferred from harmonics.
    """
    nyq = sr / 2.0
    high_hz = min(high_hz, nyq * 0.999)
    if low_hz >= high_hz:
        return x.astype(np.float32, copy=False)
    sos = signal.butter(8, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def mulaw_companding(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 mu-law encode followed by decode.

    This is a real 8-bit quantisation round trip, not a gain change: it is
    lossy, and the loss is concentrated in low-amplitude samples, which is
    exactly where fricatives and stop releases live.
    """
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    magnitude = np.log1p(mu * np.abs(x)) / np.log1p(mu)
    encoded = np.sign(x) * magnitude
    # quantise to 8 bits (256 levels across [-1, 1])
    quantised = np.round((encoded + 1.0) * 127.5)
    quantised = np.clip(quantised, 0, 255)
    decoded_norm = quantised / 127.5 - 1.0
    y = np.sign(decoded_norm) * (1.0 / mu) * ((1.0 + mu) ** np.abs(decoded_norm) - 1.0)
    return y.astype(np.float32)


def alaw_companding(x: np.ndarray, A: float = 87.6) -> np.ndarray:
    """G.711 A-law encode/decode. The European half of the G.711 pair, so
    it matters for any EU deployment."""
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    ax = np.abs(x)
    lnA = 1.0 + np.log(A)
    comp = np.where(ax < 1.0 / A, A * ax / lnA, (1.0 + np.log(np.maximum(A * ax, 1e-12))) / lnA)
    encoded = np.sign(x) * comp
    quantised = np.clip(np.round((encoded + 1.0) * 127.5), 0, 255)
    d = np.abs(quantised / 127.5 - 1.0)
    expanded = np.where(d < 1.0 / lnA, d * lnA / A, np.exp(d * lnA - 1.0) / A)
    y = np.sign(quantised / 127.5 - 1.0) * expanded
    return y.astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, kind: str = "babble",
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add noise at a measured SNR, computed over speech-active frames only.

    Measuring SNR over the whole file (silence included) is the usual bug:
    a recording with long pauses ends up far noisier than the label claims.
    """
    rng = rng or np.random.default_rng(0)
    if kind == "white":
        noise = rng.standard_normal(len(x))
    elif kind == "pink":
        white = rng.standard_normal(len(x))
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(x))
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=len(x))
    elif kind == "babble":
        # crude multi-talker babble: several band-limited, delayed noise streams
        noise = np.zeros(len(x))
        for _ in range(6):
            stream = rng.standard_normal(len(x))
            sos = signal.butter(4, [200 / 4000, 3000 / 4000], btype="band", output="sos")
            stream = signal.sosfilt(sos, stream)
            shift = rng.integers(0, max(1, len(x) // 4))
            noise += np.roll(stream, shift)
    elif kind == "hum":
        t = np.arange(len(x)) / 8000.0
        noise = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t)
    else:
        raise ValueError(f"unknown noise kind: {kind}")

    speech_power = _active_speech_power(x)
    noise_power = float(np.mean(noise ** 2)) or 1e-12
    target_noise_power = speech_power / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return (x + noise).astype(np.float32)


def _active_speech_power(x: np.ndarray, frame: int = 400, percentile: float = 70.0) -> float:
    """Mean power of the louder frames, as a stand-in for speech-active power."""
    if len(x) < frame:
        return float(np.mean(x ** 2)) or 1e-12
    n = len(x) // frame
    frames = x[: n * frame].reshape(n, frame)
    powers = np.mean(frames ** 2, axis=1)
    threshold = np.percentile(powers, percentile)
    active = powers[powers >= threshold]
    return float(np.mean(active)) if len(active) else 1e-12


def packet_loss(x: np.ndarray, sr: int, loss_rate: float,
                frame_ms: float = 20.0, conceal: str = "zero",
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Drop whole RTP-sized frames, the way a real jittery connection does.

    Loss is bursty rather than independent per frame — a Gilbert-Elliott
    two-state chain — because that is what actually happens on a bad link
    and it damages recognition far more than uniform loss at the same rate.
    """
    rng = rng or np.random.default_rng(0)
    if loss_rate <= 0:
        return x.astype(np.float32, copy=False)

    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = int(np.ceil(len(x) / frame))
    y = x.astype(np.float32).copy()

    # Gilbert-Elliott: p = good->bad, q = bad->good. Mean burst length 1/q.
    mean_burst = 3.0
    q = 1.0 / mean_burst
    p = (loss_rate * q) / max(1e-9, (1.0 - loss_rate))
    bad = False
    for i in range(n_frames):
        bad = (rng.random() > q) if bad else (rng.random() < p)
        if not bad:
            continue
        lo, hi = i * frame, min(len(y), (i + 1) * frame)
        if conceal == "zero":
            y[lo:hi] = 0.0
        elif conceal == "repeat" and lo >= (hi - lo):
            y[lo:hi] = y[lo - (hi - lo): lo]
    return y


def spectral_suppression(x: np.ndarray, sr: int, aggressiveness: float = 0.7) -> np.ndarray:
    """Spectral-subtraction noise suppression, as a stand-in for the
    aggressive NS that sits in most telephony stacks.

    Included because suppression tuned for human listeners routinely *hurts*
    recognition: it removes low-energy consonant detail and introduces
    musical noise. Being able to show that with a number is often the single
    most useful result of a run.
    """
    n_fft, hop = 512, 128
    f, t, Z = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag, phase = np.abs(Z), np.angle(Z)
    # noise floor estimated from the quietest 10% of frames per bin
    noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)
    cleaned = mag - aggressiveness * noise_mag
    floor = 0.05 * mag
    cleaned = np.maximum(cleaned, floor)
    _, y = signal.istft(cleaned * np.exp(1j * phase), fs=sr,
                        nperseg=n_fft, noverlap=n_fft - hop)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


# --------------------------------------------------------------------------
# the chain
# --------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """One reproducible channel condition.

    Serialise this into the results file so any number in a report can be
    traced back to the exact chain that produced it.
    """
    name: str = "clean"
    target_sr: int = TELEPHONY_SR
    bandlimit: bool = True
    low_hz: float = NARROWBAND_LOW_HZ
    high_hz: float = NARROWBAND_HIGH_HZ
    codec: Optional[str] = "mulaw"          # mulaw | alaw | None
    snr_db: Optional[float] = None           # None disables added noise
    noise_kind: str = "babble"
    packet_loss_rate: float = 0.0
    noise_suppression: float = 0.0           # 0 disables
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def apply_channel(x: np.ndarray, sr_in: int, cfg: ChannelConfig) -> tuple[np.ndarray, int]:
    """Run one audio array through a channel config.

    Stage order is deliberate and mirrors a real path: the microphone and the
    room add noise before the codec sees anything, suppression runs on the
    device before encoding, and packet loss happens after encoding.
    """
    rng = np.random.default_rng(cfg.seed)
    y = np.asarray(x, dtype=np.float32)

    if cfg.snr_db is not None:
        y = add_noise(y, cfg.snr_db, cfg.noise_kind, rng)

    if cfg.noise_suppression > 0:
        y = spectral_suppression(y, sr_in, cfg.noise_suppression)

    y = resample(y, sr_in, cfg.target_sr)
    sr = cfg.target_sr

    if cfg.bandlimit:
        y = bandlimit(y, sr, cfg.low_hz, cfg.high_hz)

    if cfg.codec == "mulaw":
        y = mulaw_companding(y)
    elif cfg.codec == "alaw":
        y = alaw_companding(y)
    elif cfg.codec not in (None, "none"):
        raise ValueError(f"unknown codec: {cfg.codec}")

    if cfg.packet_loss_rate > 0:
        y = packet_loss(y, sr, cfg.packet_loss_rate, rng=rng)

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32), sr


# --------------------------------------------------------------------------
# the standard grid
# --------------------------------------------------------------------------

def standard_conditions() -> list[ChannelConfig]:
    """The conditions worth reporting by default.

    Kept small on purpose. Every extra condition multiplies API spend, and
    six well-chosen points tell you more than thirty arbitrary ones.
    """
    return [
        ChannelConfig(name="wideband_clean", target_sr=16000, bandlimit=False, codec=None),
        ChannelConfig(name="telephony_clean", codec="mulaw"),
        ChannelConfig(name="telephony_alaw", codec="alaw"),
        ChannelConfig(name="telephony_noisy_15db", codec="mulaw", snr_db=15.0),
        ChannelConfig(name="telephony_noisy_5db", codec="mulaw", snr_db=5.0),
        ChannelConfig(name="telephony_lossy_3pct", codec="mulaw", packet_loss_rate=0.03),
        ChannelConfig(name="telephony_suppressed", codec="mulaw", snr_db=15.0, noise_suppression=0.8),
    ]
