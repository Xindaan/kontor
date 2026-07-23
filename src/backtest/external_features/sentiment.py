"""Sentiment engine abstractions for Phase C (T-0102).

A ``SentimentEngine`` is a small object that maps a piece of text to a
float in ``[-1, 1]`` (negative -> bearish, positive -> bullish) plus a
stable ``engine_code`` identifier. News adapters call the engine at
**pull time** to compute per-article scores; the resulting numbers go
into the long-form CSV and the headlines plus their scores land in a
``{as_of}.headlines.ndjson`` sidecar (T-0103/T-0103b).

Two production engines are supported:

- :class:`VaderSentimentEngine` — lightweight, uses NLTK's VADER
  lexicon. The lexicon must be available locally; the engine never
  performs a runtime download. If the lexicon is missing the engine
  raises with a clear setup hint.
- :class:`FinBERTSentimentEngine` — lazy import of ``transformers``/
  ``torch``; only available when the optional ``sentiment-finbert``
  Poetry group is installed.

Tests inject :class:`MockSentimentEngine` to keep CI off the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Iterable, List, Optional


class SentimentEngine(ABC):
    """Abstract sentiment engine.

    Concrete engines are deterministic per ``engine_code``; callers
    persist the code into provenance and the sidecar so that snapshots
    are reproducible and engine-drift is detectable.
    """

    @property
    @abstractmethod
    def engine_code(self) -> str:
        """Stable string id, e.g. ``vader@1.0`` or ``finbert@<model>``."""

    @property
    def engine_version(self) -> int:
        """Numeric encoding for the CSV-feature row.

        Defaults to a stable 32-bit hash of ``engine_code``; subclasses
        may override to choose a more meaningful integer.
        """

        digest = hashlib.sha256(self.engine_code.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")

    @abstractmethod
    def score(self, text: str) -> float:
        """Return a score in ``[-1, 1]`` for the input text."""

    def score_many(self, texts: Iterable[str]) -> List[float]:
        """Score a batch of texts. Default implementation loops."""

        return [self.score(t) for t in texts]


class MockSentimentEngine(SentimentEngine):
    """Deterministic, hash-based sentiment engine used by tests and the
    synthetic adapter. No external dependency."""

    _engine_code = "mock@1.0"

    @property
    def engine_code(self) -> str:
        return self._engine_code

    @property
    def engine_version(self) -> int:
        return 1

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text:
            return 0.0
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        return round(raw * 2.0 - 1.0, 6)


class VaderSentimentEngine(SentimentEngine):
    """VADER-based sentiment engine using NLTK.

    The constructor verifies that the ``vader_lexicon`` corpus is
    available locally. **No runtime download is performed**; if the
    corpus is missing the engine raises ``RuntimeError`` with a setup
    hint. CI bleibt netzwerklos.
    """

    _engine_code = "vader@1.0"

    def __init__(self) -> None:
        try:
            import nltk
        except ImportError as exc:  # pragma: no cover - install hint
            raise RuntimeError(
                "VaderSentimentEngine requires nltk. Install with "
                "`poetry install --with sentiment`."
            ) from exc
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError as exc:
            raise RuntimeError(
                "VADER lexicon not found. Run once "
                "`python -m nltk.downloader vader_lexicon`."
            ) from exc
        # Import here, after the lexicon check, to avoid touching the
        # module before we know it is usable.
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore

        self._analyzer = SentimentIntensityAnalyzer()

    @property
    def engine_code(self) -> str:
        return self._engine_code

    @property
    def engine_version(self) -> int:
        return 100  # stable numeric handle for CSV-feature drift tracking

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        scores = self._analyzer.polarity_scores(text)
        # VADER's compound score is already in [-1, 1].
        return float(scores.get("compound", 0.0))


class FinBERTSentimentEngine(SentimentEngine):
    """FinBERT-class sentiment engine via Hugging Face transformers.

    The constructor lazily imports ``transformers`` and raises with a
    setup hint if the dependency is missing. The default model is
    ``cardiffnlp/twitter-roberta-base-sentiment-latest`` — caller may
    override with a different model id.
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self, model: Optional[str] = None) -> None:
        self._model_id = model or self.DEFAULT_MODEL
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - install hint
            raise RuntimeError(
                "FinBERTSentimentEngine requires transformers + torch. "
                "Install with `poetry install --with sentiment-finbert`."
            ) from exc
        self._pipe = pipeline(
            "sentiment-analysis",
            model=self._model_id,
            top_k=None,
        )

    @property
    def engine_code(self) -> str:
        return f"finbert@{self._model_id}"

    @property
    def engine_version(self) -> int:
        return 200

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        results = self._pipe(text[:512])  # pragma: no cover - model only
        if not results:
            return 0.0
        record = results[0] if isinstance(results, list) else results
        if isinstance(record, list):  # top_k returns list-of-lists
            record = record[0]
        label = str(record.get("label", "")).lower()
        score = float(record.get("score", 0.0))
        if "neg" in label:
            return -score
        if "pos" in label:
            return score
        return 0.0


_ENGINES = {
    "mock": MockSentimentEngine,
    "vader": VaderSentimentEngine,
    "finbert": FinBERTSentimentEngine,
}


def get_sentiment_engine(name: str, **kwargs) -> SentimentEngine:
    """Factory used by adapters and CLI ``features pull --news-engine``."""

    key = (name or "").strip().lower()
    if key not in _ENGINES:
        raise ValueError(
            f"unknown sentiment engine '{name}'. Available: {sorted(_ENGINES)}"
        )
    return _ENGINES[key](**kwargs)


__all__ = [
    "FinBERTSentimentEngine",
    "MockSentimentEngine",
    "SentimentEngine",
    "VaderSentimentEngine",
    "get_sentiment_engine",
]
