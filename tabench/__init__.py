"""tabench — measure ASR accuracy on telephony-band audio."""

__version__ = "0.1.0"

from .degrade import ChannelConfig, apply_channel, standard_conditions
from .metrics import NormalizerConfig, word_errors, corpus_wer, bootstrap_ci
from .run import Manifest, Utterance, run_benchmark, save_results
from .report import build_report

__all__ = [
    "ChannelConfig", "apply_channel", "standard_conditions",
    "NormalizerConfig", "word_errors", "corpus_wer", "bootstrap_ci",
    "Manifest", "Utterance", "run_benchmark", "save_results",
    "build_report", "__version__",
]
