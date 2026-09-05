"""Turn results into something a CTO reads in ninety seconds.

Two outputs: a markdown report with the tables that matter, and a standalone
SVG heatmap. No plotting dependency — the SVG is written directly, so the
report renders anywhere including inside a GitHub README.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import ErrorCounts, NormalizerConfig, bootstrap_ci, word_errors


def _agg(rows: Iterable[dict]) -> ErrorCounts:
    total = ErrorCounts()
    for r in rows:
        total = total.merged(ErrorCounts(r["substitutions"], r["deletions"],
                                         r["insertions"], r["reference_words"]))
    return total


def _pct(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{100.0 * x:.1f}"


def _slice_table(rows: Sequence[dict], by: str, providers: Sequence[str],
                 min_words: int = 20) -> list[list[str]]:
    """WER per (slice value, provider). Slices with too few reference words
    are dropped rather than reported, because a WER over 12 words is noise."""
    values = sorted({r[by] for r in rows})
    table = [[by] + list(providers)]
    for v in values:
        line = [str(v)]
        for p in providers:
            cell = [r for r in rows if r[by] == v and r["provider"] == p]
            counts = _agg(cell)
            line.append(_pct(counts.rate) if counts.reference_length >= min_words else "-")
        if any(c != "-" for c in line[1:]):
            table.append(line)
    return table


def _md_table(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_no data_\n"
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def heatmap_svg(rows: Sequence[dict], providers: Sequence[str],
                conditions: Sequence[str]) -> str:
    """WER heatmap, condition x provider. Colour runs green->amber->red on a
    0-60% WER scale; anything above 60% is saturated because past that point
    the transcript is not usable and the exact number stops mattering."""
    cell_w, cell_h = 150, 42
    left, top = 190, 64
    width = left + cell_w * len(providers) + 24
    height = top + cell_h * len(conditions) + 44

    def colour(w: float) -> str:
        if w != w or w == float("inf"):
            return "#6b7280"
        t = max(0.0, min(1.0, w / 0.60))
        if t < 0.5:
            f = t / 0.5
            r, g, b = int(21 + f * (217 - 21)), int(128 + f * (165 - 128)), int(61 + f * (74 - 61))
        else:
            f = (t - 0.5) / 0.5
            r, g, b = int(217 + f * (185 - 217)), int(165 + f * (28 - 165)), int(74 + f * (28 - 74))
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace,Menlo,monospace">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="16" y="26" font-size="15" font-weight="600" fill="#111827">'
        f'Word error rate (%) by channel condition</text>',
        f'<text x="16" y="46" font-size="11" fill="#6b7280">'
        f'lower is better &#183; grey = no data</text>',
    ]
    for j, p in enumerate(providers):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{top - 8}" font-size="11" fill="#374151" '
                     f'text-anchor="middle">{p}</text>')
    for i, c in enumerate(conditions):
        y = top + i * cell_h
        parts.append(f'<text x="{left - 10}" y="{y + cell_h/2 + 4}" font-size="11" '
                     f'fill="#374151" text-anchor="end">{c}</text>')
        for j, p in enumerate(providers):
            cell = [r for r in rows if r["condition"] == c and r["provider"] == p]
            counts = _agg(cell)
            w = counts.rate if counts.reference_length else float("nan")
            x = left + j * cell_w
            label = _pct(w) if counts.reference_length else "-"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" '
                         f'rx="4" fill="{colour(w)}"/>')
            parts.append(f'<text x="{x + (cell_w-3)/2}" y="{y + cell_h/2 + 4}" '
                         f'font-size="13" font-weight="600" fill="#ffffff" '
                         f'text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_report(results_path: str | Path, out_dir: str | Path,
                 normalizer: NormalizerConfig = NormalizerConfig()) -> dict[str, Path]:
    with open(results_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    plan, rows = payload["plan"], payload["results"]

    # Re-score from the stored reference/hypothesis pairs rather than trusting
    # the counts written at run time. Transcription is the expensive part and
    # it is already paid for; scoring is free. So a change to text
    # normalisation costs nothing to re-evaluate, and the report stays honest
    # if the normaliser has been fixed since the run that produced the data.
    for r in rows:
        counts = word_errors(r["reference"], r["hypothesis"], normalizer)
        r["substitutions"] = counts.substitutions
        r["deletions"] = counts.deletions
        r["insertions"] = counts.insertions
        r["reference_words"] = counts.reference_length
        r["wer"] = counts.rate

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    providers = sorted({r["provider"] for r in rows})
    conditions = [c for c in plan["conditions"] if any(r["condition"] == c for r in rows)]

    baseline = "wideband_clean"
    lines: list[str] = []
    lines.append("# Telephony ASR benchmark\n")
    lines.append(f"_{plan['utterances']} utterances &#183; {len(conditions)} channel "
                 f"conditions &#183; {len(providers)} providers &#183; "
                 f"{plan['cells']} cells &#183; "
                 f"${plan.get('actual_cost_usd', 0):.2f} spent_\n")
    lines.append(f"Text normalisation: `{plan['normalizer']}`. "
                 f"Prices last checked {plan['prices_last_checked']}.\n")

    lines.append("\n## Headline\n")
    head = [["provider", "clean WER", "telephony WER", "degradation", "median RTF"]]
    for p in providers:
        base = _agg([r for r in rows if r["provider"] == p and r["condition"] == baseline])
        tel = _agg([r for r in rows if r["provider"] == p and r["condition"] == "telephony_clean"])
        rtfs = sorted(r["rtf"] for r in rows if r["provider"] == p and r["rtf"] != float("inf"))
        med = rtfs[len(rtfs)//2] if rtfs else 0.0
        delta = "-"
        if base.reference_length and tel.reference_length and base.rate > 0:
            delta = f"{tel.rate / base.rate:.1f}x worse"
        head.append([p,
                     _pct(base.rate) if base.reference_length else "-",
                     _pct(tel.rate) if tel.reference_length else "-",
                     delta, f"{med:.2f}"])
    lines.append(_md_table(head))
    lines.append("\n> The degradation column is the number worth arguing about. "
                 "A provider that looks strong on clean wideband audio and loses "
                 "half of it on an 8 kHz line is not the provider you want "
                 "answering your phone.\n")

    lines.append("\n## By channel condition\n")
    cond_tbl = [["condition"] + [f"{p} WER (95% CI)" for p in providers]]
    for c in conditions:
        line = [c]
        for p in providers:
            cell = [r for r in rows if r["condition"] == c and r["provider"] == p]
            counts = _agg(cell)
            if counts.reference_length < 20:
                line.append("-")
                continue
            lo, hi = bootstrap_ci([(r["reference"], r["hypothesis"]) for r in cell])
            line.append(f"{_pct(counts.rate)} ({_pct(lo)}&ndash;{_pct(hi)})")
        cond_tbl.append(line)
    lines.append(_md_table(cond_tbl))
    lines.append("\n> Intervals are bootstrapped over utterances. On a set this "
                 "size they are wide, and two conditions whose intervals overlap "
                 "have not been shown to differ. Quoting a WER without one is how "
                 "benchmark claims stop being falsifiable.\n")
    lines.append(f"\n![WER heatmap](wer_heatmap.svg)\n")

    for dim, title in (("accent", "By accent"), ("age_band", "By speaker age"),
                       ("domain", "By domain"), ("sex", "By speaker sex")):
        if len({r[dim] for r in rows}) > 1:
            lines.append(f"\n## {title}\n")
            lines.append(_md_table(_slice_table(rows, dim, providers)))

    lines.append("\n## Failure mode\n")
    fm = [["provider", "substitutions", "deletions", "insertions", "read"]]
    for p in providers:
        c = _agg([r for r in rows if r["provider"] == p])
        dominant = max(("mishears", c.substitutions), ("drops audio", c.deletions),
                       ("hallucinates", c.insertions), key=lambda t: t[1])[0]
        fm.append([p, str(c.substitutions), str(c.deletions), str(c.insertions), dominant])
    lines.append(_md_table(fm))
    lines.append("\n> Insertions matter out of proportion to their count. A model "
                 "that invents words on silence will happily book an appointment "
                 "nobody asked for.\n")

    errs = [r for r in rows if r.get("error")]
    if errs:
        lines.append(f"\n## Errors\n\n{len(errs)} of {len(rows)} calls failed.\n\n")
        seen = defaultdict(int)
        for r in errs:
            seen[(r["provider"], r["error"][:110])] += 1
        for (p, e), n in sorted(seen.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{p}` &#215;{n}: {e}\n")

    lines.append("\n## Reproducing this\n\n```bash\n"
                 "pip install -e .\n"
                 "tabench run --manifest <manifest.json> "
                 f"--providers {' '.join(providers)} --max-cost 5.00\n"
                 "tabench report --results out/results.json\n```\n")

    md_path = out / "REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    svg_path = out / "wer_heatmap.svg"
    svg_path.write_text(heatmap_svg(rows, providers, conditions), encoding="utf-8")
    return {"markdown": md_path, "svg": svg_path}
