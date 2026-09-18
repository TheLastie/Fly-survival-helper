"""Гибридный поиск: dense (мозг мухи) + sparse (BM25) -> RRF-фьюжн.

Почему гибрид, а не только нейросеть:
  dense отвечает на перефразы («синее небо» vs «рассеяние Рэлея»),
  sparse не пропускает редкие термины и числа (артикулы, ФИО, даты),
  которые размазываются по 512-мерному эмбеддингу.
RRF (Reciprocal Rank Fusion) объединяет ранжирования без калибровки
шкал: score = sum 1/(60 + rank) по каждому списку.
"""
from collections import Counter, defaultdict

from memory import tokenize

_BM25_K1 = 1.2
_BM25_B = 0.75
_RRF_K = 60


class SparseIndex:
    """BM25-инвертированный индекс по токенам эпизодов."""

    def __init__(self):
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)  # term -> {docid: tf}
        self.doc_len: dict[int, int] = {}
        self.avg_len: float = 1.0

    def build(self, texts: list[str]) -> "SparseIndex":
        self.postings.clear()
        self.doc_len.clear()
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.doc_len[i] = len(toks)
            tf = Counter(toks)
            for term, c in tf.items():
                self.postings[term][i] = c
        n = max(1, len(texts))
        self.avg_len = max(1.0, sum(self.doc_len.values()) / n)
        return self

    def search(self, query: str, k: int = 50):
        n_docs = max(1, len(self.doc_len))
        scores = Counter()
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = log1p((n_docs - df + 0.5) / (df + 0.5))
            for docid, tf in posting.items():
                norm = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * self.doc_len[docid] / self.avg_len)
                scores[docid] += idf * tf * (_BM25_K1 + 1) / norm
        return scores.most_common(k)


def rrf_fuse(*rank_lists, k: int = _RRF_K) -> list:
    """Списки id, упорядоченные по убыванию релевантности -> отсортированные id."""
    fused = Counter()
    for lst in rank_lists:
        for rank, docid in enumerate(lst):
            fused[docid] += 1.0 / (k + rank + 1)
    return [docid for docid, _ in fused.most_common()]


def log1p(x: float) -> float:
    import math
    return math.log1p(max(x, 1e-9))


def rm3_expand(norm_terms_fn, query_terms, top_texts, n: int = 8, min_len: int = 4):
    """Псевдо-релевантная экспансия: частотные термины топ-документов,
    не входящие в запрос (классический RM3, упрощённый)."""
    from collections import Counter
    qset = set(query_terms)
    c = Counter()
    for t in top_texts:
        c.update(norm_terms_fn(t))
    for t in qset:
        c.pop(t, None)
    return [t for t, _ in c.most_common(n * 3) if len(t) >= min_len][:n]
