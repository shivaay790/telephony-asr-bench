"""Tests. No network, no model downloads, no spend."""

import json
import math

import numpy as np
import pytest

from tabench.degrade import (ChannelConfig, add_noise, alaw_companding, apply_channel,
                             bandlimit, mulaw_companding, packet_loss, resample,
                             standard_conditions, _active_speech_power)
from tabench.metrics import (ErrorCounts, NormalizerConfig, bootstrap_ci, char_errors,
                             corpus_wer, normalize, word_errors)
from tabench.providers import MockProvider, estimate_cost, to_wav_bytes
from tabench.run import Manifest, Utterance, run_benchmark, save_results, BudgetExceeded
from tabench.report import build_report
from tabench.samples import build_sample_corpus


# ----------------------------------------------------------------- degrade

def _tone(sr=16000, secs=1.0, f=440.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_resample_changes_length_proportionally():
    x = _tone()
    y = resample(x, 16000, 8000)
    assert abs(len(y) - len(x) // 2) <= 2


def test_resample_identity_when_rates_match():
    x = _tone()
    assert np.array_equal(resample(x, 16000, 16000), x)


def test_bandlimit_attenuates_out_of_band_energy():
    sr = 8000
    low = _tone(sr, 1.0, 100.0)      # below the 300 Hz passband edge
    inband = _tone(sr, 1.0, 1000.0)
    assert np.sqrt(np.mean(bandlimit(low, sr) ** 2)) < 0.2 * np.sqrt(np.mean(low ** 2))
    assert np.sqrt(np.mean(bandlimit(inband, sr) ** 2)) > 0.5 * np.sqrt(np.mean(inband ** 2))


@pytest.mark.parametrize("companding", [mulaw_companding, alaw_companding])
def test_companding_roundtrip_is_lossy_but_close(companding):
    x = _tone()
    y = companding(x)
    assert y.shape == x.shape
    assert not np.array_equal(x, y)            # genuinely quantised
    assert np.max(np.abs(x - y)) < 0.15        # but recognisably the same signal


def test_companding_clips_to_unit_range():
    y = mulaw_companding(np.array([-4.0, 0.0, 4.0], dtype=np.float32))
    assert np.all(np.abs(y) <= 1.0 + 1e-6)


def test_add_noise_hits_requested_snr_within_a_couple_of_db():
    x = _tone()
    for target in (5.0, 15.0):
        y = add_noise(x, target, "white", np.random.default_rng(0))
        noise = y - x
        measured = 10 * math.log10(_active_speech_power(x) / np.mean(noise ** 2))
        assert abs(measured - target) < 2.0


def test_lower_snr_is_actually_noisier():
    x = _tone()
    n5 = np.mean((add_noise(x, 5.0, "white") - x) ** 2)
    n20 = np.mean((add_noise(x, 20.0, "white") - x) ** 2)
    assert n5 > n20


def test_packet_loss_zeroes_roughly_the_requested_fraction():
    x = np.ones(8000 * 4, dtype=np.float32)
    y = packet_loss(x, 8000, 0.10, rng=np.random.default_rng(1))
    lost = float(np.mean(y == 0.0))
    assert 0.03 < lost < 0.25          # bursty, so wide but bounded


def test_packet_loss_zero_rate_is_identity():
    x = _tone()
    assert np.array_equal(packet_loss(x, 16000, 0.0), x)


def test_apply_channel_resamples_and_normalises():
    x = _tone()
    y, sr = apply_channel(x, 16000, ChannelConfig(name="t", target_sr=8000))
    assert sr == 8000
    assert np.max(np.abs(y)) <= 1.0 + 1e-6


def test_channel_config_is_serialisable():
    for cfg in standard_conditions():
        assert json.loads(json.dumps(cfg.to_dict()))["name"] == cfg.name


def test_unknown_codec_raises():
    with pytest.raises(ValueError):
        apply_channel(_tone(), 16000, ChannelConfig(codec="opus"))


# ----------------------------------------------------------------- metrics

def test_normalize_strips_punctuation_and_case():
    assert normalize("Hello, World!") == "hello world"


def test_normalize_expands_number_words():
    assert normalize("i need twenty five of them") == "i need 25 of them"


def test_number_words_across_hundreds():
    assert "102" in normalize("one hundred two")
    assert "3000" in normalize("three thousand")


def test_identical_strings_have_zero_error():
    c = word_errors("the quick brown fox", "the quick brown fox")
    assert c.rate == 0.0 and c.errors == 0


def test_error_operations_are_classified():
    # reference 4 words; hypothesis drops one and changes one
    c = word_errors("the quick brown fox", "the quick red")
    assert c.reference_length == 4
    assert c.deletions >= 1 or c.substitutions >= 1
    assert c.errors == c.substitutions + c.deletions + c.insertions


def test_insertions_can_push_rate_above_one():
    c = word_errors("yes", "yes and then i would like to book an appointment tomorrow")
    assert c.rate > 1.0            # hallucination is visible, not clamped


def test_empty_hypothesis_is_all_deletions():
    c = word_errors("alpha bravo charlie", "")
    assert c.deletions == 3 and c.substitutions == 0 and c.insertions == 0


def test_spoken_digit_strings_concatenate_not_sum():
    # claim/member/phone numbers are read out digit by digit
    assert normalize("claim number four eight two") == "claim number 482"
    assert normalize("one two three") == "123"


def test_quantities_still_sum():
    assert normalize("twenty five") == "25"
    assert normalize("one hundred two") == "102"


def test_empty_reference_with_output_is_infinite_rate():
    c = word_errors("", "spurious words here")
    assert c.rate == float("inf")


def test_corpus_wer_pools_errors_not_averages_rates():
    # one long correct utterance, one short wrong one
    pairs = [("a b c d e f g h i j", "a b c d e f g h i j"), ("x", "y")]
    pooled = corpus_wer(pairs)
    assert pooled.reference_length == 11
    assert abs(pooled.rate - 1 / 11) < 1e-9    # not 0.5, which averaging gives


def test_char_errors_ignores_spacing():
    assert char_errors("hello world", "helloworld").rate == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    pairs = [("the cat sat on the mat", "the cat sat on the hat")] * 12
    lo, hi = bootstrap_ci(pairs, n_resamples=200, seed=0)
    point = corpus_wer(pairs).rate
    assert lo <= point <= hi


def test_filler_stripping_is_opt_in():
    cfg = NormalizerConfig(strip_filler=True)
    assert normalize("um i uh need help", cfg) == "i need help"
    assert "um" in normalize("um i uh need help")


# ----------------------------------------------------------------- providers

def test_wav_bytes_are_a_valid_riff_header():
    b = to_wav_bytes(_tone(8000, 0.1), 8000)
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE"


def test_cost_estimate_scales_with_conditions():
    one = estimate_cost(["deepgram"], 3600.0, 1)["TOTAL"]
    seven = estimate_cost(["deepgram"], 3600.0, 7)["TOTAL"]
    assert abs(seven - 7 * one) < 1e-9


def test_free_providers_cost_nothing():
    assert estimate_cost(["mock", "whisper_local"], 36000.0, 7)["TOTAL"] == 0.0


def test_mock_provider_degrades_with_damage():
    refs = {"u": "the quick brown fox jumps over the lazy dog again and again"}
    prov = MockProvider(refs, seed=3)
    prov.set_utterance("u")
    clean, _ = apply_channel(_tone(16000, 2.0), 16000,
                             ChannelConfig(target_sr=16000, bandlimit=False, codec=None))
    tr = prov.transcribe(clean, 16000)
    assert tr.provider == "mock" and tr.cost_usd == 0.0
    assert tr.audio_s > 0


# ----------------------------------------------------------------- end to end

def test_full_pipeline_offline(tmp_path):
    manifest = build_sample_corpus(tmp_path / "samples", n_per_group=1)
    assert len(manifest.utterances) > 0

    conditions = standard_conditions()[:2]
    results, plan = run_benchmark(manifest, ["mock"], conditions=conditions,
                                  max_cost_usd=0.0, progress=False)
    assert len(results) == len(manifest.utterances) * len(conditions)
    assert plan["actual_cost_usd"] == 0.0

    path = save_results(results, plan, tmp_path / "out")
    paths = build_report(path, tmp_path / "out")
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "Telephony ASR benchmark" in text
    assert "By channel condition" in text
    assert paths["svg"].read_text(encoding="utf-8").startswith("<svg")


def test_dry_run_sends_nothing_and_reports_a_plan(tmp_path):
    manifest = build_sample_corpus(tmp_path / "s", n_per_group=1)
    results, plan = run_benchmark(manifest, ["mock"], dry_run=True, progress=False)
    assert results == []
    assert plan["cells"] > 0
    assert "estimated_cost_usd" in plan


def test_budget_guard_refuses_before_spending(tmp_path):
    manifest = build_sample_corpus(tmp_path / "s", n_per_group=1)
    with pytest.raises(BudgetExceeded):
        run_benchmark(manifest, ["deepgram"], max_cost_usd=0.0000001, progress=False)


def test_manifest_roundtrip_resolves_relative_paths(tmp_path):
    manifest = build_sample_corpus(tmp_path / "s", n_per_group=1)
    reloaded = Manifest.load(tmp_path / "s" / "manifest.json")
    assert len(reloaded.utterances) == len(manifest.utterances)
    from pathlib import Path
    assert Path(reloaded.utterances[0].audio_path).exists()


def test_empty_manifest_is_rejected():
    with pytest.raises(ValueError):
        run_benchmark(Manifest(name="empty"), ["mock"], progress=False)
