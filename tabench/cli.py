"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .degrade import ChannelConfig, standard_conditions
from .metrics import NormalizerConfig
from .report import build_report
from .run import Manifest, BudgetExceeded, run_benchmark, save_results
from .samples import build_sample_corpus


def _conditions_from(names: list[str] | None) -> list[ChannelConfig]:
    allc = {c.name: c for c in standard_conditions()}
    if not names:
        return standard_conditions()
    missing = [n for n in names if n not in allc]
    if missing:
        raise SystemExit(f"unknown condition(s): {', '.join(missing)}\n"
                         f"available: {', '.join(allc)}")
    return [allc[n] for n in names]


def cmd_demo(args: argparse.Namespace) -> int:
    """Offline end-to-end run: generate a corpus, degrade it, score it, report.

    No API keys, no downloads, no spend. This exists so anyone who clones the
    repo sees the whole pipeline work in under a minute.
    """
    out = Path(args.out)
    print("building sample corpus (synthetic speech-like audio)...")
    manifest = build_sample_corpus(out / "samples", n_per_group=args.n)
    print(f"  {len(manifest.utterances)} utterances, "
          f"{manifest.total_audio_s:.1f}s audio")

    print("running benchmark (provider: mock)...")
    results, plan = run_benchmark(manifest, ["mock"],
                                  conditions=_conditions_from(args.conditions),
                                  max_cost_usd=0.0 if args.max_cost is None else args.max_cost)
    path = save_results(results, plan, out)
    print(f"  {len(results)} cells -> {path}")

    paths = build_report(path, out)
    print(f"report:  {paths['markdown']}")
    print(f"heatmap: {paths['svg']}")
    print("\nNote: the mock provider is a pipeline test, not a recogniser. "
          "Its WER numbers mean nothing on their own.")
    return 0


def cmd_tts_corpus(args: argparse.Namespace) -> int:
    from .tts_corpus import build_tts_corpus, TTSUnavailable
    try:
        manifest = build_tts_corpus(args.out, sentences_per_voice=args.per_voice)
    except TTSUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    path = Path(args.out) / "manifest.json"
    print(f"{len(manifest.utterances)} utterances, "
          f"{manifest.total_audio_s:.1f}s audio -> {path}")
    print("\nThis is TTS speech: cleaner than human speech, and recognisers have "
          "seen these voices.\nAbsolute WER will be optimistic. The relative "
          "degradation across channel\nconditions is the part that transfers.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    normalizer = NormalizerConfig(strip_filler=args.strip_filler)
    try:
        results, plan = run_benchmark(
            manifest, args.providers,
            conditions=_conditions_from(args.conditions),
            normalizer=normalizer,
            max_cost_usd=args.max_cost,
            dry_run=args.dry_run,
        )
    except BudgetExceeded as exc:
        print(f"\nrefusing to run: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("\ndry run — nothing was sent. Re-run without --dry-run to execute.")
        return 0

    path = save_results(results, plan, args.out)
    print(f"{len(results)} cells -> {path}")
    print(f"spent ${plan.get('actual_cost_usd', 0):.4f}")
    if not args.no_report:
        paths = build_report(path, args.out)
        print(f"report:  {paths['markdown']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    paths = build_report(args.results, args.out or Path(args.results).parent)
    print(f"report:  {paths['markdown']}")
    print(f"heatmap: {paths['svg']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tabench",
        description="Measure ASR accuracy on telephony-band audio, "
                    "not the clean wideband audio everyone benchmarks on.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="offline end-to-end run, no keys or spend")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--n", type=int, default=3, help="utterances per speaker group")
    d.add_argument("--conditions", nargs="*")
    d.add_argument("--max-cost", type=float, default=None)
    d.set_defaults(func=cmd_demo)

    t = sub.add_parser("tts-corpus",
                       help="build a real-speech corpus with the OS speech engine")
    t.add_argument("--out", default="out/tts")
    t.add_argument("--per-voice", type=int, default=3)
    t.set_defaults(func=cmd_tts_corpus)

    r = sub.add_parser("run", help="run against a manifest")
    r.add_argument("--manifest", required=True)
    r.add_argument("--providers", nargs="+", required=True,
                   help="mock whisper_local deepgram assemblyai openai")
    r.add_argument("--conditions", nargs="*")
    r.add_argument("--out", default="out")
    r.add_argument("--max-cost", type=float, default=1.0,
                   help="hard budget in USD; the run refuses to start above it")
    r.add_argument("--dry-run", action="store_true",
                   help="print the plan and cost estimate, send nothing")
    r.add_argument("--strip-filler", action="store_true")
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="rebuild the report from results.json")
    rep.add_argument("--results", required=True)
    rep.add_argument("--out", default=None)
    rep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
