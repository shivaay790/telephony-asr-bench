# telephony-asr-bench

**Speech recognition benchmarks are published on clean, wideband, close-mic audio. Voice agents don't receive that.**

They receive 8 kHz audio, band-limited to roughly 300–3400 Hz, companded to 8 bits by G.711, carried over a lossy network, and frequently pre-processed by noise suppression that was tuned for human ears rather than for a recogniser.

This measures what actually arrives.

```bash
pip install -e .
tabench demo                    # end to end, offline, no keys, no spend
```

---

## What it does

Takes a manifest of audio plus reference transcripts, pushes every utterance through a set of reproducible channel conditions, transcribes each one with every provider you name, and reports word error rate sliced by condition, accent, speaker age and domain — with bootstrapped confidence intervals, because a WER without one is not comparable to another WER.

Each stage of the channel is separately toggleable, so error can be attributed to a specific part of the chain rather than to "the phone":

| Stage | What it models |
|---|---|
| Resample to 8 kHz | The sample rate of a real call |
| Bandpass 300–3400 Hz | The POTS passband. Removes the fundamental of most adult male voices |
| μ-law / A-law | G.711 8-bit companding — a genuine lossy round trip, worst on quiet sounds |
| Additive noise | Babble, pink, white or mains hum at a measured SNR |
| Packet loss | Bursty frame loss via a Gilbert-Elliott chain, not uniform random |
| Noise suppression | Spectral subtraction, standing in for the NS in most telephony stacks |

Two details that decide whether the numbers mean anything:

- **SNR is measured over speech-active frames**, not the whole file. Measuring across silence makes a file with long pauses far noisier than its label claims.
- **Packet loss is bursty.** Real links lose runs of frames. Uniform loss at the same rate is much easier and produces flattering numbers.

## A result from the included run

Real numbers, `whisper base.en`, 12 utterances, on this machine, $0.00:

| Condition | WER % (95% CI) |
|---|---|
| Wideband clean | 4.7 (0.0–10.5) |
| Telephony, μ-law | 4.7 (0.0–10.5) |
| Telephony, A-law | 4.7 (0.0–10.5) |
| Telephony + 3% packet loss | 5.4 (0.0–12.3) |
| Telephony + noise @ 15 dB SNR | 8.7 (3.2–16.3) |
| Telephony + noise @ 15 dB + suppression | 10.7 (4.1–18.8) |
| Telephony + noise @ 5 dB SNR | 32.2 (17.9–45.7) |

**Band-limiting to telephony costs nothing here.** Going from 16 kHz wideband to 8 kHz μ-law moved WER not at all — nor did A-law. That is worth knowing, because "we're on 8 kHz" is the usual first explanation for a bad number, and on this evidence it is the wrong one. Noise is what hurts: roughly 7× degradation at 5 dB SNR, and that interval does not come near the clean one.

**The suppression result is suggestive and not established.** Suppression looks worse than no suppression on the same noise — 10.7 against 8.7 — which is the direction you would predict, since spectral subtraction removes exactly the low-energy consonant detail a recogniser depends on. But at n=12 the intervals overlap heavily. **That is not a finding.** It is a hypothesis worth 200 utterances, and the tool says so rather than letting you quote it. This is the whole point of printing intervals: the first draft of this README claimed the effect outright, and the confidence interval is what caught it.

**Scoring is separate from transcription.** `tabench report` recomputes error rates from the stored transcripts, so changing text normalisation costs nothing. That matters more than it sounds: a normalisation bug found later would otherwise mean re-paying for every API call.

## Usage

```bash
# offline: synthetic audio, mock provider, exercises the whole pipeline
tabench demo

# real speech from your OS speech engine, then a real recogniser, still $0
tabench tts-corpus --per-voice 3
tabench run --manifest out/tts/manifest.json --providers whisper_local --max-cost 0

# hosted providers — always dry-run first
tabench run --manifest my.json --providers deepgram assemblyai --dry-run
tabench run --manifest my.json --providers deepgram assemblyai --max-cost 2.50

# rebuild the report from saved results
tabench report --results out/results.json
```

### Spend control

Cell count is multiplicative — utterances × conditions × providers — and easy to misjudge. So:

- `--dry-run` prints the plan and a per-provider cost estimate and **sends nothing**.
- `--max-cost` is a hard pre-flight guard. If the estimate exceeds it the run **refuses to start**, before a single request.
- Spend is tracked during the run and it stops if the budget is reached mid-flight.
- Prices live in one dict in `providers.py` with the date last checked, and every report prints the prices it used.

```
$ tabench run --manifest my.json --providers deepgram openai --dry-run
{
  "utterances": 200, "conditions": [...7...], "cells": 2800,
  "estimated_cost_usd": {"deepgram": 1.81, "openai": 2.52, "TOTAL": 4.33},
  "prices_last_checked": "2026-09-05"
}
```

## Providers

| Key | Cost | Notes |
|---|---|---|
| `mock` | free | Deterministic pipeline test. Not a recogniser; its WER means nothing alone |
| `whisper_local` | free | faster-whisper or openai-whisper. Sweep parameters here before spending |
| `deepgram` | metered | `nova-2-phonecall` by default |
| `assemblyai` | metered | Upload, poll, retrieve |
| `openai` | metered | `whisper-1` |

Hosted providers read `DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY`, `OPENAI_API_KEY`. A missing key is recorded on the result rather than raised, so one unavailable provider doesn't abort a run. Network failures retry with backoff and are counted in the report — how often a provider simply failed to answer is itself a result.

Adding a provider is one class with a `transcribe(audio, sr) -> Transcript` method; see `providers.py`.

## Your own data

Point it at a manifest:

```json
{
  "name": "clinic-intake-sample",
  "license": "describe it here",
  "utterances": [
    {
      "id": "utt_001",
      "audio_path": "audio/utt_001.wav",
      "reference": "I need to reschedule my appointment with doctor Chen",
      "accent": "indian", "age_band": "sixties_plus",
      "sex": "female", "domain": "clinical"
    }
  ]
}
```

Metadata is the point. An aggregate WER tells you nothing actionable; *"WER is three times higher for speakers over sixty"* is a roadmap item. See [docs/DATASETS.md](docs/DATASETS.md) for corpora that ship speaker demographics.

## Honesty about the bundled corpora

- `tabench demo` uses **synthetic formant audio**. It is not speech and cannot be transcribed. It exists so the pipeline runs anywhere in under a minute.
- `tabench tts-corpus` uses your **OS speech engine**. That is real, recognisable speech with a known transcript — but it is cleaner than human speech, has no disfluency, overlap, room or emotion, and recognisers have almost certainly seen these voice families. **Absolute WER from it is optimistic and must not be quoted as a provider's real accuracy.** What transfers is the relative degradation between conditions on identical source audio.

For numbers you would put in front of someone, use a real corpus.

## Ethics

Build evaluation sets from public, licensed speech corpora. Do not use recordings of real patients or customers. If you are measuring a live product, use its public demo path, keep volume to what research requires, and send named results privately before publishing anything.

## Install

```bash
pip install -e .              # core: numpy, scipy, soundfile
pip install -e ".[http]"      # + hosted providers
pip install -e ".[local]"     # + faster-whisper
pip install -e ".[dev]"       # + pytest
python -m pytest              # 20 tests, no network
```

## Licence

MIT. Built by [Shivaay Dhondiyal](https://shivaaydhondiyal.online/consulting) — first author on [FlowFake](https://shivaaydhondiyal.online/flowfake.pdf) (ICML 2026 ML for Audio workshop), previously voice-agent workflows and call analytics at 1M+ calls/month.
