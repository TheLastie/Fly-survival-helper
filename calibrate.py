"""Автокалибровка порогов и уверенности под реальный корпус (этап P0).

Генерирует пробы из самого корпуса (окна текста как «цитаты-пользователя»),
меряет распределение знакомости известного vs невиданного, подбирает:
  theta_low — гейт знакомости;
  calib_a/b — conf = sigmoid(a·dense_cos + b);
  calib_a2/b2 — conf для BM25-only кандидатов по нормированному BM25-скору.

Запуск: python calibrate.py <корпус> [--size 3] [--overlap 1] [--probes 150]
"""
import argparse
import json
import os
import re
import time

import numpy as np

from calibration import fit_logistic
from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem

_WORD = re.compile(r"[a-zа-я0-9]+", re.I)


def make_probe(text: str, rng) -> str | None:
    """Окно 6-10 слов из середины чанка — имитация цитаты/перефраза."""
    toks = _WORD.findall(text)
    if len(toks) < 14:
        return None
    start = int(rng.integers(1, len(toks) - 10))
    n = int(rng.integers(6, 11))
    return " ".join(toks[start:start + n])


def make_novel(rng) -> str:
    """Вопрос из случайных слов, гарантированно отсутствующих в корпусе."""
    soup = ["квандор", "мелтар", "сювен", "драксель", "торвин", "альмег",
            "призмунд", "хаварта", "тюльпанарий", "космоглиф"]
    a, b = rng.choice(soup, size=2, replace=False)
    tpl = ["о чём говорится в %s и как связано с %s",
           "кто такой %s и почему он %s",
           "расскажи про %s против %s",
           "как %s повлиял на %s"]
    return str(rng.choice(tpl)) % (a, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--probes", type=int, default=150)
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--state", default=None, help="сохранить состояние индекса")
    args = ap.parse_args()

    sys_ = FlySystem(FlyConfig(decay=0.0), HashingEmbedder())
    t0 = time.time()
    r = sys_.index_path(args.corpus, size=args.size, overlap=args.overlap)
    print(f"индексировано: {r['added']} эпизодов, {r['dups']} дублей, "
          f"{time.time()-t0:.1f} с")
    n = len(sys_.memory)
    if n < 50:
        print("корпус слишком мал для калибровки (<50 эпизодов)")
        return 1

    rng = np.random.default_rng(7)
    ids = rng.choice(n, size=min(args.probes, n), replace=False)

    dense_raw, bm_raw, labels, bm_labels = [], [], [], []
    fam_known = []
    for i in ids:
        gold_text = sys_.memory.texts[i]
        q = make_probe(gold_text, rng)
        if q is None:
            continue
        x = sys_.embedder.embed(q)
        fam_known.append(sys_.brain.familiarity(x))
        # dense top-1
        dtop = sys_.memory.topk(x, 1)
        dense_raw.append(dtop[0][1] if dtop else 0.0)
        labels.append(int(dtop and dtop[0][0] == i))
        # bm25 top-1 (нормированный)
        hits = sys_.store.search_bm25(q, 5)
        bm_raw.append(hits[0][1] if hits else 0.0)
        bm_labels.append(int(bool(hits) and hits[0][0] == i))

    fam_novel = [sys_.brain.familiarity(sys_.embedder.embed(make_novel(rng)))
                 for _ in range(len(fam_known))]
    med_k, med_n = float(np.median(fam_known)), float(np.median(fam_novel))
    theta_low = float(np.clip((med_k + med_n) / 2, 0.05, 0.45))

    a, b = fit_logistic(dense_raw, labels)
    bm_max = max(bm_raw) if bm_raw else 1.0
    bm_norm = [s / bm_max for s in bm_raw]
    a2, b2 = fit_logistic(bm_norm, labels, a0=4.0, b0=-2.0)
    hit_rate = float(np.mean(labels)) if labels else 0.0
    bm_hit = float(np.mean(bm_labels)) if bm_labels else 0.0

    calib = {"theta_low": round(theta_low, 3),
             "calib_a": round(a, 2), "calib_b": round(b, 2),
             "calib_a2": round(a2, 2), "calib_b2": round(b2, 2),
             "stats": {"probes": len(labels), "dense_hit@1": round(hit_rate, 3),
                       "bm_hit@1": round(bm_hit, 3),
                       "fam_known_med": round(med_k, 3),
                       "fam_novel_med": round(med_n, 3)}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"\ncalib -> {args.out}: {calib}")
    if args.state:
        t0 = time.time()
        sys_.save(args.state)
        print(f"состояние сохранено: {args.state} ({time.time()-t0:.1f} с)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
