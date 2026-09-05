"""Build a real-speech corpus locally with the OS text-to-speech engine.

Why this exists: the synthetic corpus in `samples.py` exercises the pipeline
but cannot be transcribed, so it cannot produce a real WER. Downloading a
speech corpus is the right answer for publishable numbers, but it needs
bandwidth, disk and a licence conversation before anyone can see the tool
work.

This module sits in between. It drives the operating system's own TTS to
produce genuine, recognisable speech with a known reference transcript, so a
real recogniser produces real error rates on a laptop with no network.

WHAT THIS IS NOT
TTS speech is easier than human speech: no disfluency, no overlapping talk,
no room, no emotion, and a recogniser has almost certainly seen this voice
family in training. Absolute WER here will be optimistic and must never be
quoted as a provider's real-world accuracy. What survives is the *relative*
degradation between channel conditions measured on identical source audio,
which is the question this benchmark exists to answer.

Backends: Windows SAPI via PowerShell, macOS `say`, Linux `espeak-ng`.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from .run import Manifest, Utterance
from .samples import SENTENCES


class TTSUnavailable(RuntimeError):
    pass


# (voice hint, rate, label used for the accent slice in the report)
WINDOWS_VOICES = [
    ("David", 0, "sapi_david"),
    ("Zira", 0, "sapi_zira"),
    ("David", -3, "sapi_david_slow"),
    ("Zira", 3, "sapi_zira_fast"),
]
MAC_VOICES = [("Alex", 175, "say_alex"), ("Samantha", 175, "say_samantha"),
              ("Daniel", 150, "say_daniel"), ("Karen", 200, "say_karen")]
LINUX_VOICES = [("en-us", 150, "espeak_us"), ("en-gb", 150, "espeak_gb"),
                ("en-us+f3", 150, "espeak_us_f"), ("en-gb", 190, "espeak_gb_fast")]


def _synth_windows(text: str, voice: str, rate: int, out_wav: Path) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -like '*{voice}*' }} | Select-Object -First 1; "
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
        f"$s.Rate = {rate}; "
        f"$s.SetOutputToWaveFile('{out_wav}'); "
        f"$s.Speak('{text}'); "
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   check=True, capture_output=True, timeout=120)


def _synth_mac(text: str, voice: str, rate: int, out_wav: Path) -> None:
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                   check=True, capture_output=True, timeout=120)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                    str(aiff), str(out_wav)], check=True, capture_output=True)
    aiff.unlink(missing_ok=True)


def _synth_linux(text: str, voice: str, rate: int, out_wav: Path) -> None:
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(rate),
                    "-w", str(out_wav), text],
                   check=True, capture_output=True, timeout=120)


def _backend():
    system = platform.system()
    if system == "Windows":
        return _synth_windows, WINDOWS_VOICES
    if system == "Darwin" and shutil.which("say"):
        return _synth_mac, MAC_VOICES
    if shutil.which("espeak-ng"):
        return _synth_linux, LINUX_VOICES
    raise TTSUnavailable(
        "No local TTS backend found.\n"
        "  Windows: built in, nothing to install\n"
        "  macOS:   built in (`say`)\n"
        "  Linux:   sudo apt install espeak-ng\n"
        "Or skip this and point --manifest at a real speech corpus; "
        "see docs/DATASETS.md."
    )


def build_tts_corpus(out_dir: str | Path, sentences_per_voice: int = 3,
                     target_sr: int = 16000) -> Manifest:
    """Synthesise a small real-speech corpus and write a manifest.

    Idempotent: existing WAVs are reused, so re-running is cheap.
    """
    synth, voices = _backend()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(
        name="tts-local",
        license="generated locally by the OS speech engine; not redistributable as a corpus",
        source="tabench.tts_corpus.build_tts_corpus",
    )

    flat: list[tuple[str, str]] = []
    for domain, lines in SENTENCES.items():
        for s in lines:
            flat.append((domain, s))

    idx = 0
    for voice, rate, label in voices:
        for k in range(sentences_per_voice):
            domain, text = flat[idx % len(flat)]
            idx += 1
            uid = f"{label}_{domain}_{k}"
            path = out / f"{uid}.wav"

            if not path.exists():
                with tempfile.TemporaryDirectory() as td:
                    raw = Path(td) / "raw.wav"
                    try:
                        synth(text, voice, rate, raw)
                    except subprocess.CalledProcessError as exc:
                        raise TTSUnavailable(
                            f"TTS backend failed for voice {voice!r}: "
                            f"{exc.stderr.decode(errors='ignore')[:200]}"
                        ) from exc
                    audio, sr = sf.read(raw, dtype="float32", always_2d=False)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != target_sr:
                        from .degrade import resample
                        audio = resample(audio, sr, target_sr)
                    sf.write(path, audio, target_sr, subtype="PCM_16")

            info = sf.info(str(path))
            manifest.utterances.append(Utterance(
                id=uid, audio_path=path.name, reference=text,
                accent=label, age_band="unknown", sex="unknown",
                domain=domain, duration_s=info.frames / info.samplerate,
            ))

    manifest_path = out / "manifest.json"
    manifest.save(manifest_path)
    return Manifest.load(manifest_path)
