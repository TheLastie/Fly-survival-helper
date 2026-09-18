"""Линейный LTR-ранкер на признаках + pairwise-обучение на DAN-фидбеке (E5).

Идея: RRF даёт разнообразную кандидатную выборку, но финальный порядок —
одномерный. Переранжировщик скорит кандидатов линейной комбинацией
признаков; веса обучаются pairwise-логистической потерей на событиях
журнала: «верно» (winner) против показанных, но не подтверждённых (losers),
аналогично «забудь».

Защита от переобучения: признак обновляется только после min_pairs пар
(иначе остаётся на дефолте) — DAN-фидбек разрежён.
"""
import numpy as np

FEATURES = ["bm25", "bm25_head", "dense", "dense_nbr", "coverage",
            "len_norm", "rare_hit", "dan"]
N_FEAT = len(FEATURES)
# разумные дефолты до обучения
DEFAULT_W = np.array([1.0, 0.6, 1.0, 0.5, 0.8, 0.3, 0.4, 0.4], dtype=np.float64)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LinearRanker:
    def __init__(self, min_pairs: int = 20):
        self.w = DEFAULT_W.copy()
        self.min_pairs = min_pairs
        self.pair_count = np.zeros(N_FEAT, dtype=np.int64)
        self.n_pairs = 0

    def score(self, feats: np.ndarray) -> float:
        return float(self.w @ feats)

    def fit_pairs(self, pairs, lr: float = 0.05, lam: float = 1e-4,
                  epochs: int = 3) -> int:
        """pairs: [(f_winner, f_loser), ...]. Онлайн SGD по log-loss.
        Возвращает число использованных пар."""
        used = 0
        for _ in range(epochs):
            for fw, fl in pairs:
                d = fw - fl
                p = sigmoid(float(self.w @ d))
                grad = (1.0 - p) * d - lam * self.w
                # заморозка: признак обновляется только после min_pairs пар,
                # в которых он реально различался
                upd = self.pair_count >= self.min_pairs
                self.pair_count += (np.abs(d) > 1e-9).astype(np.int64)
                self.w += lr * grad * upd
                used += 1
        self.n_pairs += used
        return used

    def state_dict(self):
        return {"w": self.w, "pair_count": self.pair_count,
                "n_pairs": self.n_pairs}

    def load_state_dict(self, st) -> None:
        self.w = np.asarray(st["w"], dtype=np.float64)
        self.pair_count = np.asarray(st["pair_count"], dtype=np.int64)
        self.n_pairs = int(st["n_pairs"])


# ------------------------------------------------------- feature extraction

def extract_features(query_terms: list, bm25_raw: float, dense_raw: float,
                     text: str, heading: str | None, doc_terms: set,
                     rare_terms: set, dl: int, avgdl: float,
                     nbr_dense: float, dan: float) -> np.ndarray:
    """Вектор признаков f1..f8 (см. FEATURES). Все компоненты ~[0,1]."""
    bm = bm25_raw / (bm25_raw + 5.0)                       # мягкая нормировка
    head_terms = set()
    if heading:
        from textnorm import norm_terms
        head_terms = set(norm_terms(heading))
    bh_raw = _bm25_like(query_terms, head_terms)
    bh = bh_raw / (bh_raw + 2.0)
    den = max(0.0, dense_raw)
    cov = len(set(query_terms) & doc_terms) / max(1, len(set(query_terms)))
    lnorm = 1.0 / (1.0 + dl / max(1.0, avgdl))
    rare = 1.0 if (rare_terms & doc_terms) else 0.0
    return np.array([bm, bh, den, max(0.0, nbr_dense), cov, lnorm, rare, dan],
                    dtype=np.float64)


def _bm25_like(query_terms, doc_terms) -> float:
    if not doc_terms:
        return 0.0
    hit = sum(1 for t in set(query_terms) if t in doc_terms)
    return hit / max(1.0, np.sqrt(len(set(query_terms))))


if __name__ == "__main__":
    # самопроверка: обучение на синтетических парах двигает вес dense вверх
    r = LinearRanker(min_pairs=2)
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(60):
        f_pos = np.array([0.5, 0.1, 0.8, 0.3, 0.7, 0.2, 1.0, 0.0])
        f_neg = np.array([0.5, 0.1, 0.3, 0.3, 0.7, 0.2, 0.0, 0.0])
        np_pos = rng.normal(0, 0.02, N_FEAT)
        np_neg = rng.normal(0, 0.02, N_FEAT)
        d7 = np_pos[7] - np_neg[7]            # разность по dim 7 обнуляем:
        np_neg[7] += d7                       # в парах dan не различается
        pairs.append((f_pos + np_pos, f_neg + np_neg))
    w0 = r.w.copy()
    r.fit_pairs(pairs)
    assert r.w[2] > w0[2] + 0.1, "dense-признак должен вырасти"
    assert r.w[6] > w0[6], "rare_hit должен вырасти"
    assert abs(r.w[7] - w0[7]) < 1e-9, "неактивный признак (dan) не трогаем"
    print("ranker: самопроверка пройдена, w =", np.round(r.w, 2))
