# Corpora worth using

The bundled corpora exist so the tool runs anywhere in a minute. Neither produces numbers you should show anyone. For that you need real recorded speech, and — if you want the slices that make the report actionable — speaker metadata.

## The short answer

**Mozilla Common Voice.** CC0, tens of languages, and it is the only widely used corpus that ships **speaker age and accent** as standard fields. Those two columns are what turn "our WER is 11%" into "our WER triples for speakers over sixty," which is the sentence that changes a roadmap.

## Options

| Corpus | Licence | Metadata | Good for |
|---|---|---|---|
| [Common Voice](https://commonvoice.mozilla.org/datasets) | CC0 | age, accent, sex, locale | Accent and age slices. Start here |
| [LibriSpeech](https://www.openslr.org/12) | CC BY 4.0 | sex only | A clean, universally comparable baseline |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | CC0 | language, some speaker id | European accented English at scale |
| [TED-LIUM 3](https://www.openslr.org/51/) | CC BY-NC-ND | speaker id | Spontaneous speech; **non-commercial only** |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | CC BY 4.0 | speaker id, roles | Overlapping speech, far-field, diarization |
| [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) | LDC, paid | rich | Genuine 8 kHz telephone speech. The real thing |
| [Fisher English](https://catalog.ldc.upenn.edu/LDC2004S13) | LDC, paid | rich | Large-scale conversational telephone speech |

Read the licence before you publish a number. TED-LIUM is `NC-ND`, so it cannot back a commercial claim. Switchboard and Fisher are the only ones here that are *actually* telephone speech rather than wideband audio you degraded yourself — if you are making a public claim about telephony, one of them belongs in the mix, and the fact that they cost money is the reason so few published benchmarks include them.

## Building a manifest from Common Voice

Common Voice ships a TSV plus MP3s. This is enough:

```python
import csv, json
from pathlib import Path

CV = Path("cv-corpus-XX.0-en")
rows = list(csv.DictReader(open(CV / "validated.tsv", encoding="utf-8"), delimiter="\t"))

# Keep only clips whose speaker metadata is actually filled in — most is not,
# and a slice built from blanks is worse than no slice at all.
rows = [r for r in rows if r.get("age") and r.get("accents")][:400]

manifest = {
    "name": "common-voice-en-sample",
    "license": "CC0",
    "source": "https://commonvoice.mozilla.org/datasets",
    "utterances": [
        {
            "id": r["path"].replace(".mp3", ""),
            "audio_path": str((CV / "clips" / r["path"]).resolve()),
            "reference": r["sentence"],
            "accent": (r.get("accents") or "unknown").split(",")[0].strip(),
            "age_band": r.get("age") or "unknown",
            "sex": r.get("gender") or "unknown",
            "domain": "general",
        }
        for r in rows
    ],
}
json.dump(manifest, open("cv_manifest.json", "w"), indent=2)
```

Then:

```bash
tabench run --manifest cv_manifest.json --providers whisper_local --max-cost 0
tabench run --manifest cv_manifest.json --providers deepgram --dry-run
```

`soundfile` reads MP3 via libsndfile 1.1+. On an older build, convert once:
`ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`.

## Sizing a run

Two numbers decide everything:

- **Cells** = utterances × conditions × providers. 400 × 7 × 3 is 8,400 API calls. Always `--dry-run` first.
- **Reference words per slice.** A slice under ~500 reference words has a confidence interval wide enough to hide any effect you care about. The report drops slices under 20 words entirely rather than printing noise, but 20 is a floor for *displaying* a number, not a bar for believing it.

If budget is tight: cut conditions before you cut utterances. Four well-chosen conditions across 400 utterances beats seven conditions across 100.

## Domain vocabulary

The `domain` field exists because general STT fails in specific, predictable places: drug names, procedure codes, menu items, member IDs, street names. If you are benchmarking for a clinic or a drive-thru, a general corpus will flatter every provider you test. Record or synthesise a few hundred in-domain utterances and tag them — that slice is usually where the actionable result lives.

## Ethics, again

Do not build eval sets from recordings of real patients or customers. Consent for a service call is not consent to be a benchmark. Public corpora exist precisely so this is not necessary.
