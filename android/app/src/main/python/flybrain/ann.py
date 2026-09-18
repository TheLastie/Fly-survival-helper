"""IVF-ANN (E4): инвертированный файловый индекс по центроидам.

Вместо точного матвектора по всем M документам (O(M·D) на запрос):
  1) k-means по подвыборке -> nlist центроидов (одноразово при finalize);
  2) каждый вектор приписан ближайшему центроиду -> списки docid;
  3) запрос: топ-nprobe центроидов -> точный dot только по их спискам.

Recall контролируется nprobe: 8 быстро, 32 — почти точный поиск.
Это не PQ (коды не режутся): память те же float16-вектора + ~10 Б/док.
"""
import numpy as np


class IVF:
    def __init__(self, nlist: int = 256, nprobe: int = 16, seed: int = 0,
                 iters: int = 12, sample_max: int = 65536):
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.iters = iters
        self.sample_max = sample_max
        self.cent = None       # (nlist, D) float32, нормированы
        self.ids = None        # docids, отсортированы по assign
        self.indptr = None     # (nlist+1,) границы списков
        self.assign_count = None
        self.vecs = None       # ссылка на float16 матрицу
        self.n = 0

    def fit(self, vecs: np.ndarray) -> "IVF":
        self.vecs = vecs
        self.n = len(vecs)
        rng = np.random.default_rng(self.seed)
        if self.n > self.sample_max:
            sub = vecs[rng.choice(self.n, self.sample_max, replace=False)]
        else:
            sub = vecs
        X = sub.astype(np.float32)
        nlist = min(self.nlist, max(16, self.n // 64))
        cent = X[rng.choice(len(X), nlist, replace=False)].copy()
        for _ in range(self.iters):
            d = X @ cent.T
            assign = d.argmax(axis=1)
            for j in range(nlist):
                m = assign == j
                if m.any():
                    cent[j] = X[m].mean(axis=0)
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        self.cent = cent.astype(np.float32)
        # приписываем все векторы
        A = vecs.astype(np.float32) @ self.cent.T
        self.assign_count = np.bincount(A.argmax(axis=1), minlength=nlist)
        order = np.argsort(A.argmax(axis=1), kind="stable")
        self.ids = order.astype(np.int32)
        self.indptr = np.zeros(nlist + 1, dtype=np.int64)
        np.cumsum(self.assign_count, out=self.indptr[1:])
        return self

    def query(self, vec: np.ndarray, k: int, nprobe: int | None = None):
        """Топ-k (docid, cos) среди nprobe ближайших списков."""
        if self.cent is None or self.n == 0:
            return []
        np_ = nprobe or self.nprobe
        d = self.cent @ vec
        lists = np.argsort(-d)[:min(np_, self.nlist)]
        parts = [self.ids[self.indptr[j]:self.indptr[j + 1]] for j in lists]
        ids = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        if len(ids) == 0:
            return []
        sims = self.vecs[ids].astype(np.float32) @ vec
        k = min(k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def recall_check(self, vecs, exact_topk_fn, k: int = 10, n: int = 200,
                     seed: int = 1) -> float:
        """Доля точного top-k, перекрытого ANN top-k (на случайных запросах)."""
        rng = np.random.default_rng(seed)
        hits = 0
        for i in rng.choice(len(vecs), size=min(n, len(vecs)), replace=False):
            v = vecs[i].astype(np.float32)
            ann = {j for j, _ in self.query(v, k)}
            exact = set(exact_topk_fn(v, k))
            hits += len(ann & exact) / max(1, len(exact))
        return hits / min(n, len(vecs))


if __name__ == "__main__":
    # кластерные данные (как текстовые эмбеддинги), а не случайные:
    # ближайшие соседи реально живут рядом
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(64, 256)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = centers[rng.integers(64, size=20000)] + 0.05 * rng.normal(
        size=(20000, 256)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ivf = IVF(nlist=128, nprobe=16).fit(X.astype(np.float16))

    def exact(v, k):
        s = X @ v
        return set(np.argsort(-s)[:k].tolist())

    r10 = ivf.recall_check(X.astype(np.float16), exact, k=10)
    assert r10 >= 0.9, f"recall too low: {r10}"
    print(f"IVF self-check: recall@10 = {r10:.3f} OK")
