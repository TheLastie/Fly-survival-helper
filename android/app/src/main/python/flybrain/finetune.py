"""Fine-tune головы распознавания: линейный пробник поверх замороженных
MobileNet-эмбеддингов (512-d, после QR-проекции).

Обучается на всех собранных базах (leafsnap_db/fungi_db/plants_db/nature_db),
оценивается на реальных holdout'ах (LeafSnap 203 фото, грибы 24). Сравнение
с NNN-путём recognize() — цифры решают, идёт ли пробник в прод.

Запуск: python finetune.py [--db ../leafsnap_db ../fungi_db ../plants_db ../nature_db]
"""
import argparse
import glob
import os
import tempfile

import numpy as np

CLASSES_CACHE = "probe_classes.json"


def collect(db_roots):
    """[(path, label)] по всем папкам видов (jpg в папках с card.md)."""
    items = []
    for root in db_roots:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            sp = os.path.basename(d.rstrip("/"))
            for p in sorted(glob.glob(d + "*.jpg") + glob.glob(d + "*.jpeg")):
                items.append((p, sp))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+",
                    default=["../leafsnap_db", "../fungi_db",
                             "../plants_db", "../nature_db"])
    ap.add_argument("--holdout-leafsnap", default="../leafsnap_holdout")
    ap.add_argument("--holdout-fungi", default="../fungi_holdout")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", default="probe_model.npz")
    args = ap.parse_args()

    from image_embedder import ImageEmbedder
    emb = ImageEmbedder()

    items = collect(args.db)
    species = sorted({sp for _, sp in items})
    sp2id = {s: i for i, s in enumerate(species)}
    print(f"обучение: {len(items)} фото, {len(species)} классов")

    X = emb.embed_batch([p for p, _ in items])
    y = np.array([sp2id[sp] for _, sp in items], dtype=np.int64)

    # мультиномиальная лог. регрессия (numpy SGD, L2)
    n, d, C = len(X), X.shape[1], len(species)
    W = np.zeros((C, d), dtype=np.float64)
    b = np.zeros(C, dtype=np.float64)
    X64 = X.astype(np.float64)
    rng = np.random.default_rng(0)
    lr = 1.0
    for it in range(args.iters):
        idx = rng.integers(0, n, size=min(512, n))
        z = X64[idx] @ W.T + b                      # (bs, C)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        p[np.arange(len(idx)), y[idx]] -= 1.0
        g = p / len(idx)
        W -= lr * (g.T @ X64[idx] + 1e-4 * W)
        b -= lr * g.sum(axis=0)
        lr *= 0.997

    def probe_eval(root, name):
        gold_dirs = sorted(glob.glob(os.path.join(root, "*/")))
        ok = n_tot = 0
        conf_sum = 0.0
        for gd in gold_dirs:
            gold = os.path.basename(gd.rstrip("/"))
            for p in sorted(glob.glob(gd + "*.jpg"))[:4]:
                v = emb.embed_image_tta(p).astype(np.float64)
                z = W @ v + b
                z -= z.max()
                e = np.exp(z)
                pr = e / e.sum()
                pred = int(pr.argmax())
                conf_sum += float(pr.max())
                n_tot += 1
                ok += int(species[pred] == gold)
        print(f"{name}: top-1 {ok}/{n_tot} ({ok/max(1,n_tot):.3f}), "
              f"средняя уверенность {conf_sum/max(1,n_tot):.3f}")

    print("\n== пробник ==")
    probe_eval(args.holdout_leafsnap, "leafsnap holdout")
    probe_eval(args.holdout_fungi, "fungi holdout")

    # сохранение
    np.savez_compressed(args.out, W=W.astype(np.float32), b=b.astype(np.float32),
                        classes=np.array(species, dtype=object))
    print(f"\nмодель -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
