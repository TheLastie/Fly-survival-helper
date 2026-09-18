"""Калибровка порога recognize() по размеченным hold-out фото.

Сценарий: база видов проиндексирована (index_species по --db), есть папка
с тестовыми снимками, разложенными по видам: holdout/<Вид>/*.jpg (снимки,
НЕ входившие в индексацию). Для каждого кандидатного порога считаем
precision/recall/F1 вердикта recognize. Рекомендация — минимальный порог
с precision >= --min-precision (безопасность: лучше ложный отказ, чем
ложный «съедобно»), максимизирующий recall.

Результат: recognizer_calib.json {"recognize_threshold": t} — система
подхватывает его автоматически (механизм _load_calib).

Запуск: python calibrate_recognize.py --db виды/ --holdout holdout/
"""
import argparse
import glob
import json
import os

import numpy as np

from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="корень базы видов (виды/<Вид>/)")
    ap.add_argument("--holdout", required=True, help="holdout/<Вид>/*.jpg")
    ap.add_argument("--min-precision", type=float, default=0.97)
    ap.add_argument("--out", default="recognizer_calib.json")
    args = ap.parse_args()

    sys_ = FlySystem(FlyConfig(), HashingEmbedder())
    r = sys_.index_species(args.db)
    print(f"индексировано: {r['species']} видов, {r['images']} фото")

    probes = []  # (path, gold_species)
    for sp_dir in sorted(glob.glob(os.path.join(args.holdout, "*/"))):
        gold = os.path.basename(sp_dir.rstrip("/"))
        for p in sorted(glob.glob(sp_dir + "*.jpg") + glob.glob(sp_dir + "*.jpeg")
                        + glob.glob(sp_dir + "*.png")):
            probes.append((p, gold))
    if not probes:
        print("holdout пуст — нечего калибровать")
        return 1

    # верхний cos по золотому виду и по лучшему конкуренту для каждого снимка
    rows = []
    for p, gold in probes:
        q = sys_._img_embedder().embed_image_tta(p)
        hits = sys_.store.search_dense(q, 40)
        gold_cos, best_other = 0.0, 0.0
        for idx, cos in hits:
            m = sys_.memory.meta[idx] or {}
            if m.get("kind") != "image":
                continue
            if m.get("species") == gold:
                gold_cos = max(gold_cos, cos)
            else:
                best_other = max(best_other, cos)
        rows.append((gold_cos, best_other, gold_cos >= best_other))

    # матрица путаницы: gold -> предсказанный вид при рабочем пороге
    oper_t = json.load(open(args.out)).get("recognize_threshold", 0.55) \
        if os.path.exists(args.out) else 0.55
    confusion: dict = {}
    for p, gold in probes:
        out = sys_.recognize(p, threshold=oper_t)
        pred = out["verdict"] or "НЕ_РАСПОЗНАНО"
        if pred != gold:
            key = f"{gold} -> {pred}"
            confusion[key] = confusion.get(key, 0) + 1
    if confusion:
        print(f"\nпутаницы (порог {oper_t}):")
        for k, v in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {k}: {v}")
    else:
        print(f"\nпутаниц нет (порог {oper_t})")

    thresholds = np.round(np.arange(0.30, 0.951, 0.05), 2)
    print(f"\n{'порог':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'вердиктов':>9}")
    best_t, best_rec = None, -1.0
    for t in thresholds:
        tp = sum(1 for g, _, _ in rows if g >= t)
        fp = sum(1 for g, o, _ in rows if o >= t and g < t) + \
             sum(1 for g, o, gold_wins in rows if g >= t and not gold_wins)
        fn = sum(1 for g, _, _ in rows if g < t)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"{t:>6.2f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {tp:>9}")
        if prec >= args.min_precision and rec > best_rec:
            best_t, best_rec = float(t), rec
    if best_t is None:
        print(f"\nни один порог не даёт precision>={args.min_precision} — "
              "база недостаточна/путает виды")
        return 1
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"recognize_threshold": round(best_t, 2)}, f)
    print(f"\nрекомендация: recognize_threshold={best_t:.2f} "
          f"(recall={best_rec:.3f}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
