"""Weighted soft token matcher (BERTScore-style) on BGE embeddings.

Model: BAAI/bge-base-en-v1.5. Contextual token vectors
are pooled to whole words, then each reference word is matched to its most similar
candidate word by cosine similarity. Per-word weights (manual labels in [0, 1],
0 = unrelated to the message's meaning, 1 = complete match/important) scale each
word's contribution; stop words / filler are dropped.

    recall    = sum_i w_i * max_j cos(r_i, c_j) / sum_i w_i     (advisor words covered)
    precision = sum_j w_j * max_i cos(r_i, c_j) / sum_j w_j     (model words supported)
    score     = F1(precision, recall)

Requires: pip install torch transformers numpy matplotlib
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

STOP_WORDS = frozenset(
    "a an the and or but if of to in on at by for with as is are was were be been being "
    "it its this that these those i me my we our you your he she they them their so than "
    "then there here also can could should would will may might do does did have has had "
    "from into about over such very just".split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-+#.]*")


def words(text: str) -> list[tuple[str, int, int]]:
    """Word, char_start, char_end. This is the unit that gets a manual weight."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


@dataclass
class MatchResult:
    precision: float
    recall: float
    f1: float
    # per-word best-match similarity for the reference (advisor) side; feed to a histogram
    ref_words: list[str] = field(default_factory=list)
    ref_sims: list[float] = field(default_factory=list)
    ref_weights: list[float] = field(default_factory=list)


class TokenMatcher:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def _embed_words(self, text: str) -> tuple[list[str], np.ndarray]:
        """Contextual embedding per whole word (mean of its word-piece vectors)."""
        ws = words(text)
        enc = self.tok(
            text, return_offsets_mapping=True, return_tensors="pt",
            truncation=True, max_length=512,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        hidden = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[0]
        hidden = torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy()

        kept, vecs = [], []
        for w, s, e in ws:
            idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
            if idx:  # words past the 512-token truncation have no pieces and are skipped
                kept.append(w)
                vecs.append(hidden[idx].mean(0))
        v = np.stack(vecs) if vecs else np.zeros((0, hidden.shape[1]))
        return kept, v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def _weights(ws: list[str], manual: list[float] | None) -> np.ndarray:
        """Manual labels win; otherwise 1.0 for content words, 0.0 for filler."""
        if manual is not None:
            if len(manual) != len(ws):
                raise ValueError(f"{len(manual)} weights for {len(ws)} words")
            w = np.asarray(manual, dtype=float)
        else:
            w = np.array([0.0 if x.lower() in STOP_WORDS else 1.0 for x in ws])
        return w

    def score(
        self,
        reference: str,
        candidate: str,
        ref_weights: list[float] | None = None,
        cand_weights: list[float] | None = None,
    ) -> MatchResult:
        rw, rv = self._embed_words(reference)
        cw, cv = self._embed_words(candidate)
        wr = self._weights(rw, ref_weights)
        wc = self._weights(cw, cand_weights)
        if not len(rw) or not len(cw) or wr.sum() == 0 or wc.sum() == 0:
            return MatchResult(0.0, 0.0, 0.0)

        sim = rv @ cv.T                      # (ref words, cand words) cosine matrix
        r_best = sim.max(axis=1)             # best candidate match for each advisor word
        c_best = sim.max(axis=0)             # best advisor match for each model word
        recall = float((wr * r_best).sum() / wr.sum())
        precision = float((wc * c_best).sum() / wc.sum())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return MatchResult(precision, recall, f1, rw, r_best.tolist(), wr.tolist())


def labeling_template(text: str) -> list[list]:
    """[[word, default_weight], ...] to hand-edit and save into the labels JSON."""
    return [[w, 0.0 if w.lower() in STOP_WORDS else 1.0] for w, _, _ in words(text)]


def load_weights(path: str, stage: str, source: str) -> list[float] | None:
    """Labels file: {stage: {"advisor": [[word, w], ...], "gemini": [...], "openai": [...]}}"""
    try:
        with open(path) as f:
            return [w for _, w in json.load(f)[stage][source]]
    except (FileNotFoundError, KeyError):
        return None


def plot_distributions(results: dict[str, dict[str, MatchResult]], out_path: str) -> None:
    """results[stage][model] -> MatchResult. One weighted similarity histogram per stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = list(results)
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 3.5), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 21)
    for ax, stage in zip(axes[0], stages):
        for model, r in results[stage].items():
            ax.hist(r.ref_sims, bins=bins, weights=r.ref_weights, alpha=0.55,
                    label=f"{model} (F1 {r.f1:.2f})")
        ax.set_title(stage)
        ax.set_xlabel("advisor-word match similarity")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("weighted word count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
