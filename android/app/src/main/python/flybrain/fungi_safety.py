"""Второй контур безопасности: бинарный классификатор «ядовито ли?».

Датасет Kaggle "edible and poisonous fungi" бинарный (4 папки без видов) —
не подходит для видового распознавания, но идеален для жизненно важного
вопроса «это вообще можно есть?». Обучаем логистическую регрессию поверх
ImageEmbedder-эмбеддингов (512-d): P(ядовито) используется в recognize()
как ВЕТО — даже уверенный видовой вердикт гасится предупреждением.

Запуск:
  python fungi_safety.py --zip ../edible-and-poisonous-fungi.zip \
      --per-class 250 --out fungi_safety.npz
"""
import argparse
import os
import tempfile
import zipfile

import numpy as np

EDIBLE = ("edible mushroom sporocarp", "edible sporocarp")
POISON = ("poisonous mushroom sporocarp", "poisonous sporocarp")


def extract_subset(zf: zipfile.ZipFile, per_class: int, root: str):
    """До per_class jpg из каждой группы -> [(path, y)]."""
    from collections import defaultdict
    got = defaultdict(list)
    for n in zf.namelist():
        if not n.lower().endswith((".jpg", ".jpeg")):
            continue
        top = n.split("/")[0]
        y = 1 if top in POISON else (0 if top in EDIBLE else None)
        if y is None:
            continue
        for g in (EDIBLE if y == 0 else POISON):
            if top == g and len(got[g]) < per_class:
                got[g].append(n)
                break
    paths = []
    for g in EDIBLE + POISON:
        y = 1 if g in POISON else 0
        for n in sorted(got[g]):
            dst = os.path.join(root, str(y), os.path.basename(n))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(zf.read(n))
            paths.append((dst, y))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="../edible-and-poisonous-fungi.zip")
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--out", default="fungi_safety.npz")
    args = ap.parse_args()

    from image_embedder import ImageEmbedder
    zf = zipfile.ZipFile(args.zip)
    with tempfile.TemporaryDirectory() as td:
        paths = extract_subset(zf, args.per_class, td)
        print(f"извлечено {len(paths)} фото")
        emb = ImageEmbedder()
        X = emb.embed_batch([p for p, _ in paths])
        y = np.array([yy for _, yy in paths], dtype=np.float64)

        # 80/20 стратифицированный сплит
        rng = np.random.default_rng(7)
        idx0 = np.flatnonzero(y == 0)
        idx1 = np.flatnonzero(y == 1)
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        tr = np.concatenate([idx0[: int(len(idx0) * 0.8)], idx1[: int(len(idx1) * 0.8)]])
        te = np.concatenate([idx0[int(len(idx0) * 0.8):], idx1[int(len(idx1) * 0.8):]])

        # логистическая регрессия (numpy SGD, L2)
        w = np.zeros(X.shape[1], dtype=np.float64)
        b = 0.0
        X64 = X.astype(np.float64)
        for _ in range(300):
            z = X64[tr] @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y[tr])
            w -= 0.5 * (X64[tr].T @ g / len(tr) + 1e-3 * w)
            b -= 0.5 * float(g.mean())

        prob = 1.0 / (1.0 + np.exp(-np.clip(X64[te] @ w + b, -30, 30)))
        pred = prob >= 0.5
        yt = y[te]
        tp = int((pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum())
        fn = int((~pred & (yt == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        acc = float((pred == yt).mean())
        print(f"test: acc={acc:.3f} precision(ядовито)={prec:.3f} "
              f"recall(ядовито)={rec:.3f} (n={len(te)})")
        # срабатывание вето на уровне P>=0.3 (осторожность)
        veto = prob >= 0.3
        print(f"вето P>=0.3: покрывает {100*float((veto & (yt==1)).sum())/max(1,(yt==1).sum()):.1f}% "
              f"ядовитых, ложных тревог {int((veto & (yt==0)).sum())}")
        np.savez_compressed(args.out, w=w.astype(np.float32), b=np.float32(b))
        print(f"модель -> {args.out} (веса {w.nbytes//1024} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
