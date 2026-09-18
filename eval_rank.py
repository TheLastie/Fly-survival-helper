"""Офлайн-оценка качества ранжирования (этап E7).

Два источника метрик:
 1. Автопробы (окна текста как «цитаты», gold = исходный чанк) —
    полный пайплайн search() как его видит пользователь:
    Recall@1/5/10, MRR, nDCG@10. Плюс ablation: dense-only и BM25-only
    (кто что вкладывает в итоговую выдачу).
 2. Журнал DAN-фидбека — «пользовательское» качество:
    pairwise-accuracy ранкера (picked выше прочих shown),
    позиция picked в свежей выдаче.

Запуск: python eval_rank.py --state book_state [--probes 150]
"""
import argparse
import json

import numpy as np

from calibrate import make_probe
from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem


def dcg(ranks):
    return sum(1.0 / np.log2(r + 2) for r in ranks)


def eval_probes(sys_, n_probes: int, seed: int = 7) -> dict:
    n = len(sys_.memory)
    rng = np.random.default_rng(seed)
    ids = rng.choice(n, size=min(n_probes, n), replace=False)
    pos_full, pos_dense, pos_bm = [], [], []
    for i in ids:
        q = make_probe(sys_.memory.texts[i], rng)
        if q is None:
            continue
        x = sys_.embedder.embed(q)
        res = sys_.search(q, k=10)
        order = [c.idx for c in res.candidates]
        pos_full.append(order.index(i) + 1 if i in order else 99)
        dtop = [j for j, _ in sys_.memory.topk(x, 10)]
        pos_dense.append(dtop.index(i) + 1 if i in dtop else 99)
        btop = [j for j, _ in sys_.store.search_bm25(q, 10)]
        pos_bm.append(btop.index(i) + 1 if i in btop else 99)

    def metrics(pos):
        pos = np.array(pos, dtype=float)
        return {
            "R@1": round(float((pos <= 1).mean()), 3),
            "R@5": round(float((pos <= 5).mean()), 3),
            "R@10": round(float((pos <= 10).mean()), 3),
            "MRR": round(float((1.0 / pos).mean()), 3),
            # один релевантный док на позиции p: nDCG = 1/log2(p+1), p>10 -> 0
            "nDCG@10": round(float(np.mean([1.0 / np.log2(p + 1)
                                            if p <= 10 else 0.0 for p in pos])), 3),
        }
    return {"full": metrics(pos_full), "dense_only": metrics(pos_dense),
            "bm25_only": metrics(pos_bm), "probes": len(pos_full)}


def eval_journal(sys_) -> dict:
    if not sys_.journal:
        return {"events": 0}
    from textnorm import norm_terms
    pair_ok = pair_n = 0
    picked_pos = []
    for ev in sys_.journal:
        if len(ev["shown"]) < 2:
            continue
        x = sys_.embedder.embed(ev["q"])
        bm = dict(sys_.store.search_bm25(ev["q"], 50))
        dense = dict(sys_.memory.topk(x, 50))
        qt = sys_.store.query_terms(ev["q"])
        scored = []
        for idx in ev["shown"]:
            if idx >= len(sys_.memory):
                continue
            f = sys_._features_from(qt, idx, x, dense.get(idx, 0.0),
                                    bm.get(idx, 0.0))
            scored.append((idx, sys_.ranker.score(f)))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda t: t[1], reverse=True)
        top_idx = scored[0][0]
        pair_n += 1
        pair_ok += int(top_idx == ev["picked"])
        res = sys_.search(ev["q"], k=5)
        order = [c.idx for c in res.candidates]
        picked_pos.append(order.index(ev["picked"]) + 1
                          if ev["picked"] in order else 99)
    return {"events": pair_n,
            "pairwise_acc": round(pair_ok / max(1, pair_n), 3),
            "picked_pos": picked_pos}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="book_state")
    ap.add_argument("--probes", type=int, default=150)
    args = ap.parse_args()

    sys_ = FlySystem(FlyConfig(decay=0.0), HashingEmbedder())
    sys_.load(args.state)
    print(f"состояние: {len(sys_.memory)} эпизодов, "
          f"journal={len(sys_.journal)}, ranker pairs={sys_.ranker.n_pairs}")

    rep = eval_probes(sys_, args.probes)
    print("\n== автопробы (gold-чанк) ==")
    for name in ("full", "dense_only", "bm25_only"):
        m = rep[name]
        print(f"{name:10s} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} "
              f"R@10={m['R@10']:.3f} MRR={m['MRR']:.3f} nDCG@10={m['nDCG@10']:.3f}")
    print(f"probes={rep['probes']}")

    jr = eval_journal(sys_)
    print("\n== журнал DAN ==")
    print(json.dumps(jr, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
