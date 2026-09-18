"""Сборка готового field_state для телефона: индексирует все базы один раз
здесь, телефон только копирует состояние (загрузка ~секунды вместо
индексации ~минут).

Запуск: python build_phone_bundle.py --dbs ../plants_db ../fungi_db \
          ../poisonous_db ../nature_db --out phone_field_state
"""
import argparse
import os

from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", required=True)
    ap.add_argument("--out", default="phone_field_state")
    args = ap.parse_args()

    sys_ = FlySystem(FlyConfig(decay=0.0), HashingEmbedder())
    for d in args.dbs:
        if not os.path.isdir(d):
            print(f"(пропущено: {d})")
            continue
        r = sys_.index_species(d)
        print(f"{d}: {r['species']} видов, {r['images']} фото, {r['cards']} карточек")
    sys_.save(args.out)
    n_img = sum(1 for m in sys_.memory.meta if m and m.get("kind") == "image")
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fns in os.walk(args.out) for f in fns) / 1e6
    print(f"\nбандл -> {args.out}/: {len(sys_.memory)} документов, "
          f"{n_img} фото, {len(sys_.species_info)} видов, {sz:.1f} МБ")
    print("на телефоне: положите папку рядом с flybrain и запустите "
          "python flybrain/cli.py --load " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
