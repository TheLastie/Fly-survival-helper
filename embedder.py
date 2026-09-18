"""Эмбеддеры.

По умолчанию — HashingEmbedder: мешок токенов + символьных n-грамм,
хэшируется в d_model измерений. Работает полностью офлайн, без моделей.

Если установлен sentence-transformers, можно использовать
SentenceTransformerEmbedder (multilingual-MiniLM, 384-d) — для него
калибровку и пороги нужно подобрать через capacity.py.
"""
import hashlib
import re

import numpy as np


class HashingEmbedder:
    """Офлайн-эмбеддер: бинарный хэш признаков со знаком + SIF/PCA (E3).

    Два режима:
      — до fit(): равномерный мешок признаков (прежнее поведение);
      — после fit(corpus): веса SIF w(t)=a/(a+p(t)) по частотам самого
        корпуса и вычитание первой главной компоненты (power iteration),
        убирающей «тематический шум».
    """

    def __init__(self, d_model: int = 512, sif_a: float = 1e-3, pc_weight: float = 0.25):
        self.d_model = d_model
        self.sif_a = sif_a
        self.pc_weight = pc_weight
        self.ngram_weight = 0.3
        self._freq: dict | None = None
        self._total = 0
        self._pc: np.ndarray | None = None
        self._n_fit = 0

    # ---------- fit ----------

    def fit(self, texts, max_sample: int = 20000, seed: int = 0) -> None:
        """Оценить частоты терминов корпуса и первую главную компоненту."""
        from collections import Counter
        from textnorm import norm_terms
        cnt = Counter()
        for t in texts:
            cnt.update(norm_terms(t))
        self._freq = dict(cnt)
        self._total = max(1, sum(cnt.values()))
        self._n_fit = len(texts)
        if len(texts) < 8:
            self._pc = None
            return
        rng = np.random.default_rng(seed)
        if len(texts) > max_sample:
            idx = rng.choice(len(texts), size=max_sample, replace=False)
            sample = [texts[i] for i in idx]
        else:
            sample = list(texts)
        X = np.stack([self._embed_weighted(t) for t in sample]).astype(np.float32)
        # power iteration для первой компоненты (без материализации X^T X)
        v = rng.normal(size=self.d_model).astype(np.float32)
        v /= np.linalg.norm(v)
        for _ in range(30):
            u = X.T @ (X @ v)
            n = float(np.linalg.norm(u))
            if n < 1e-12:
                break
            v = (u / n).astype(np.float32)
        self._pc = v

    def state_dict(self):
        return {"freq": self._freq, "total": self._total,
                "pc": self._pc, "n_fit": self._n_fit}

    def load_state_dict(self, state) -> None:
        self._freq = state.get("freq")
        self._total = int(state.get("total", 0))
        self._pc = state.get("pc")
        self._n_fit = int(state.get("n_fit", 0))

    # ---------- эмбеддинг ----------

    def _feats(self, text: str):
        t = text.lower().replace("ё", "е")
        toks = re.findall(r"[a-zа-я0-9]+", t)
        feats = [(x, 1.0) for x in toks]
        s = " " + " ".join(toks) + " "
        for n in (3, 4):  # символьные n-граммы — устойчивость к словоформам
            for i in range(len(s) - n + 1):
                feats.append(("#" + s[i:i + n] + "#", self.ngram_weight))
        return feats

    def _w(self, term: str) -> float:
        if not self._freq or not self._total:
            return 1.0
        p = self._freq.get(term, 0) / self._total
        return self.sif_a / (self.sif_a + p) if p > 0 else 1.0

    def _embed_weighted(self, text: str) -> np.ndarray:
        from textnorm import stem
        v = np.zeros(self.d_model, dtype=np.float32)
        for f, w in self._feats(text):
            if w == 1.0 and f[0] != "#":
                w = self._w(stem(f))
            h = int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "little")
            idx = h % self.d_model
            v[idx] += w if (h >> 32) & 1 else -w
        return v

    def embed(self, text: str) -> np.ndarray:
        v = self._embed_weighted(text)
        if self._pc is not None:
            v = v - self.pc_weight * float(v @ self._pc) * self._pc
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class SentenceTransformerEmbedder:
    """Эмбеддер на sentence-transformers (нужен pip-пакет и скачивание модели)."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.d_model = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        e = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(e[0], dtype=np.float32)


def make_embedder(kind: str = "hash", **kw):
    if kind == "st":
        return SentenceTransformerEmbedder(**kw)
    return HashingEmbedder(**kw)
