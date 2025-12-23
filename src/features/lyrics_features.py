from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class LyricsVectorizer:
    vectorizer: TfidfVectorizer

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        return x.astype(np.float32).toarray()


def fit_tfidf(
    texts: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> tuple[np.ndarray, LyricsVectorizer]:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        strip_accents=None,
        lowercase=True,
    )
    x = vec.fit_transform(texts)
    return x.astype(np.float32).toarray(), LyricsVectorizer(vectorizer=vec)
