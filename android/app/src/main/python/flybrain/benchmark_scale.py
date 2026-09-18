"""Замер масштабных пределов текущей точной схемы (предварительный шаг E4).

Генерирует синтетический корпус из M уникальных фактов, индексирует,
меряет: время индексации, размер состояния, RAM поискового индекса,
латентность запроса p50/p95 (полный search), R@1/R@10 на цитатных пробах.

По результатам решаем, нужен ли IVF-PQ (E4): порог — p95 > 50 мс
или RAM матрицы > 1.5 ГБ.

Запуск: python benchmark_scale.py --m 30000 [--probes 150]
"""
import argparse
import os
import tempfile
import time

import numpy as np

from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem

NAMES = ["Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Попов",
         "Волков", "Соколов", "Лебедев", "Козлов", "Новиков", "Морозов"]
ROLES = ["инженер", "биолог", "стажер", "математик", "техник", "аналитик"]
LABS = ["Альфа", "Бета", "Гамма", "Дельта", "Омега", "Вега"]
PROJS = ["радар", "штатив", "калибр", "меридиан", "полюс", "кварц", "град"]
VERBS = ["разработал", "описал", "изучил", "проверил", "уточнил", "измерил"]
TOPICS = ["датчик", "спектр", "сигнал", "образец", "протокол", "механизм"]


def make_corpus(m: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    facts = []
    for i in range(m):
        f = (f"{NAMES[rng.integers(len(NAMES))]} {ROLES[rng.integers(len(ROLES))]} "
             f"лаборатории {LABS[rng.integers(len(LABS))]} {VERBS[rng.integers(len(VERBS))]} "
             f"{TOPICS[rng.integers(len(TOPICS))]} проекта {PROJS[rng.integers(len(PROJ := PROJS))]} "
             f"в серии опытов {i}.")
        facts.append(f)
    return facts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=30000)
    ap.add_argument("--probes", type=int, default=150)
    args = ap.parse_args()

    facts = make_corpus(args.m)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "corpus.txt")
        open(path, "w", encoding="utf-8").write("\n".join(facts))

        cfg = FlyConfig(decay=0.0)
        sys_ = FlySystem(cfg, HashingEmbedder())
        t0 = time.time()
        r = sys_.index_path(path, size=1, overlap=0)
        t_index = time.time() - t0
        n = len(sys_.memory)

        t0 = time.time()
        state = os.path.join(td, "state")
        sys_.save(state)
        t_save = time.time() - t0
        size_mb = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fns in os.walk(state) for f in fns) / 1e6

        rng = np.random.default_rng(11)
        ids = rng.choice(n, size=min(args.probes, n), replace=False)
        lats, r1, r10 = [], 0, 0
        for i in ids:
            toks = facts[i].split()
            q = " ".join(toks[2:9])
            t0 = time.time()
            res = sys_.search(q, k=10)
            lats.append((time.time() - t0) * 1000)
            order = [c.idx for c in res.candidates]
            r1 += int(order and order[0] == i)
            r10 += int(i in order)
        lats = np.array(lats)

        mat_mb = sys_.memory.matrix().nbytes / 1e6
        print(f"M={n} чанков")
        print(f"  индексация: {t_index:.1f} с ({n/max(t_index,1e-9):.0f} чанков/с)")
        print(f"  save: {t_save:.1f} с, состояние {size_mb:.1f} МБ")
        print(f"  RAM матрицы dense: {mat_mb:.0f} МБ (+ postings CSR "
              f"{sys_.store.pdocs.nbytes/1e6:.0f} МБ)")
        print(f"  запрос p50={np.percentile(lats,50):.1f} мс "
              f"p95={np.percentile(lats,95):.1f} мс max={lats.max():.1f} мс")
        print(f"  цитатные пробы: R@1={r1/len(ids):.3f} R@10={r10/len(ids):.3f}")
        per_doc_kb = size_mb * 1024 / n
        print(f"  вывод: ~{per_doc_kb:.2f} КБ/чанк на диске -> "
              f"лимит 400 МБ ~= {int(400/per_doc_kb*1000)} чанков")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
