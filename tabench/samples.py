"""A synthetic corpus so the demo runs offline.

This is deliberately *not* speech. It is a formant-style synthesiser that
produces audio with roughly speech-like spectral structure, so the degradation
chain, the scoring, the slicing and the report can all be exercised with no
downloads and no licence questions.

For real numbers, point the runner at a real corpus. `docs/DATASETS.md`
covers Common Voice (CC0, and the only widely-used set that ships speaker age
and accent metadata) and how to build a manifest from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .run import Manifest, Utterance

SR = 16000

# Sentences chosen to look like the traffic these systems actually carry:
# an appointment line, a drive-thru, and an insurance call.
SENTENCES = {
    "clinical": [
        "I need to reschedule my appointment with doctor Chen to next Tuesday morning",
        "my daughter has had a fever of one hundred and two since Thursday",
        "can you confirm whether my referral to the cardiology department went through",
        "I am calling about the prescription refill for my blood pressure medication",
    ],
    "food_order": [
        "can I get two cheeseburgers no onions and a large fries",
        "I want the number three combo with a diet coke and extra napkins",
        "do you have anything without dairy in it for my son",
        "make that three tacos and a side of rice and beans please",
    ],
    "insurance": [
        "I am checking the status of prior authorization for claim number four eight two",
        "can you tell me whether this member is still active under the group plan",
        "the claim was submitted on the fifteenth and we have not heard anything back",
        "I need the deductible and the out of pocket maximum for this policy",
    ],
}

# Formant centres (Hz) loosely following adult speaker ranges. The point is
# that the groups differ spectrally, so slicing the report by group shows
# something rather than noise.
VOICE_PROFILES = {
    ("general_american", "thirties", "female"): dict(f0=210, formants=(730, 2090, 2850)),
    ("general_american", "thirties", "male"):   dict(f0=120, formants=(660, 1720, 2410)),
    ("indian", "twenties", "male"):             dict(f0=135, formants=(620, 1660, 2500)),
    ("spanish", "forties", "female"):           dict(f0=200, formants=(700, 2000, 2760)),
    ("general_american", "sixties_plus", "male"): dict(f0=105, formants=(600, 1580, 2300)),
}


def _synth_utterance(text: str, profile: dict, seed: int) -> np.ndarray:
    """One pseudo-utterance: a syllable train with formant structure,
    amplitude envelope, and pauses roughly where a speaker would take them."""
    rng = np.random.default_rng(seed)
    syllables = max(4, len(text.split()) * 2)
    out = []

    for i in range(syllables):
        dur = rng.uniform(0.09, 0.20)
        n = int(SR * dur)
        t = np.arange(n) / SR

        # glottal source: harmonic stack with a natural -12 dB/octave rolloff
        f0 = profile["f0"] * rng.uniform(0.92, 1.08)
        src = np.zeros(n)
        for h in range(1, 26):
            if f0 * h >= SR / 2:
                break
            src += (1.0 / h ** 1.6) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))

        # formant resonances
        shaped = np.zeros(n)
        for fc, bw in zip(profile["formants"], (80, 110, 160)):
            from scipy import signal as _sig
            sos = _sig.butter(2, [max(50, fc - bw) / (SR / 2),
                                  min(SR / 2 - 1, fc + bw) / (SR / 2)],
                              btype="band", output="sos")
            shaped += _sig.sosfilt(sos, src)

        # a fricative burst on some syllables: broadband, quiet, and the first
        # thing an 8 kHz channel destroys
        if rng.random() < 0.35:
            burst = rng.standard_normal(n) * 0.05
            from scipy import signal as _sig
            sos = _sig.butter(4, 3000 / (SR / 2), btype="high", output="sos")
            shaped += _sig.sosfilt(sos, burst)

        env = np.hanning(n) ** 0.5
        out.append(shaped * env)

        if rng.random() < 0.2:
            out.append(np.zeros(int(SR * rng.uniform(0.05, 0.18))))

    audio = np.concatenate(out).astype(np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.7).astype(np.float32)


def build_sample_corpus(out_dir: str | Path, n_per_group: int = 3) -> Manifest:
    """Write WAVs plus a manifest. Idempotent — safe to re-run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        name="synthetic-demo",
        license="CC0 (generated, not recorded speech)",
        source="tabench.samples.build_sample_corpus",
    )

    seed = 0
    for (accent, age, sex), profile in VOICE_PROFILES.items():
        for domain, sentences in SENTENCES.items():
            for k in range(n_per_group):
                text = sentences[(seed + k) % len(sentences)]
                uid = f"{accent}_{age}_{sex}_{domain}_{k}"
                path = out / f"{uid}.wav"
                if not path.exists():
                    audio = _synth_utterance(text, profile, seed)
                    sf.write(path, audio, SR, subtype="PCM_16")
                info = sf.info(str(path))
                manifest.utterances.append(Utterance(
                    id=uid, audio_path=path.name, reference=text,
                    accent=accent, age_band=age, sex=sex, domain=domain,
                    duration_s=info.frames / info.samplerate,
                ))
                seed += 1

    manifest_path = out / "manifest.json"
    manifest.save(manifest_path)
    # reload so audio paths are absolute, exactly as a user's run would be
    return Manifest.load(manifest_path)
