"""Тест распознавания видов по фото (мультимодальный контур).

Корпус из двух тестовых изображений sklearn (+варианты как отдельные
«снимки» вида): «Ромашка луговая» (flower.jpg) и «Гранитный массив»
(china.jpg). Проверяется: фото->вид (top-1, уверенность), отказ при
неуверенности (шум), совет из карточки, предупреждение о ядовитости.

Запуск: python recognizetest.py
"""
import os
import tempfile

import numpy as np

from config import FlyConfig
from embedder import HashingEmbedder
from PIL import Image
from sklearn.datasets import load_sample_images
from system import FlySystem

results = []


def check(name, cond, detail=""):
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main() -> int:
    ds = load_sample_images()
    china = Image.fromarray(ds.images[0]).convert("RGB")
    flower = Image.fromarray(ds.images[1]).convert("RGB")

    with tempfile.TemporaryDirectory() as root:
        # --- вид 1: Ромашка (нетоксична) ---
        sp1 = os.path.join(root, "Ромашка луговая")
        os.makedirs(sp1)
        flower.save(os.path.join(sp1, "a.jpg"))
        flower.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(sp1, "b.jpg"))
        flower.rotate(8).save(os.path.join(sp1, "c.jpg"))
        open(os.path.join(sp1, "card.md"), "w", encoding="utf-8").write(
            "# Ромашка луговая\n"
            "ядовит: нет\n"
            "Ромашка луговая — белые лепестки и жёлтая сердцевина. "
            "Цветёт с мая по сентябрь на лугах и опушках.\n"
            "Настой цветков ромашки применяют при расстройстве желудка. "
            "В выживании ромашка полезна как противовоспалительное средство.\n")
        # --- вид 2: Гранитный массив (не растение — проверка разделения) ---
        sp2 = os.path.join(root, "Гранитный массив")
        os.makedirs(sp2)
        china.save(os.path.join(sp2, "a.jpg"))
        china.rotate(-5).save(os.path.join(sp2, "b.jpg"))
        open(os.path.join(sp2, "card.md"), "w", encoding="utf-8").write(
            "# Гранитный массив\n"
            "ядовит: нет\n"
            "Гранитный массив — скальное образование. У основания возможны "
            "трещины с почвой. В выживании гранит даёт укрытие и ориентир.\n")

        sys_ = FlySystem(FlyConfig(), HashingEmbedder())
        r = sys_.index_species(root)
        print(f"индексация видов: {r}")
        check("виды проиндексированы", r["images"] == 5 and r["cards"] >= 4,
              f"images={r['images']}, cards={r['cards']}")
        check("реестр токсичности",
              sys_.species_info.get("Ромашка луговая", {}).get("toxic") is False)

        # запрос: новый «снимок» ромашки (яркость+поворот)
        probe = os.path.join(root, "probe.jpg")
        flower.rotate(4).point(lambda v: min(255, int(v * 1.05))).save(probe)
        out = sys_.recognize(probe)
        print("распознавание:", {k: v for k, v in out.items() if k != "advice"})
        check("фото -> верный вид", out["verdict"] == "Ромашка луговая")
        check("уверенность достаточна", out["certain"] and out["matches"][0]["cos"] > 0.7,
              f"cos={out['matches'][0]['cos'] if out['matches'] else 0}")
        check("совет из карточки", "advice" in out and "ромашк" in out["advice"].lower(),
              (out.get("advice") or "")[:60])

        # неуверенный случай: белый шум
        noise = Image.fromarray(
            np.random.default_rng(5).integers(0, 255, (300, 300, 3), np.uint8))
        noise_path = os.path.join(root, "noise.jpg")
        noise.save(noise_path)
        out2 = sys_.recognize(noise_path)
        check("шум -> отказ (не уверен)", not out2["certain"] and out2["verdict"] is None,
              f"top cos={out2['matches'][0]['cos'] if out2['matches'] else 0}")

        # ядовитый вид: предупреждение в совете
        sp3 = os.path.join(root, "Воронец")
        os.makedirs(sp3)
        china.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(sp3, "a.jpg"))
        open(os.path.join(sp3, "card.md"), "w", encoding="utf-8").write(
            "# Воронец\nядовит: да\nВоронец — ядовитое растение. "
            "Симптомы отравления: тошнота.\n")
        sys_.index_species(root)
        out3 = sys_.recognize(os.path.join(sp3, "a.jpg"))
        check("ядовитый вид: предупреждение",
              out3["verdict"] == "Воронец" and "ЯДОВИТ" in (out3.get("advice") or ""),
              (out3.get("advice") or "")[:50])

        # margin-правило: два почти идентичных вида -> cos>=порог, но
        # отрыв < recognize_margin -> вердикт НЕ выдаётся (безопасность)
        spn1, spn2 = os.path.join(root, "Близнец А"), os.path.join(root, "Близнец Б")
        os.makedirs(spn1); os.makedirs(spn2)
        china.save(os.path.join(spn1, "a.jpg"))
        china.rotate(1).save(os.path.join(spn2, "a.jpg"))  # почти та же фотография
        for d, nm in ((spn1, "Близнец А"), (spn2, "Близнец Б")):
            open(os.path.join(d, "card.md"), "w", encoding="utf-8").write(
                f"# {nm}\nядовит: да\nВид {nm}, отличие едва заметно.\n")
        sys_.index_species(root)
        out_m = sys_.recognize(os.path.join(spn2, "a.jpg"))
        check("margin: близнецы -> отказ (cos>=порог, отрыв мал)",
              not out_m["certain"] and out_m["verdict"] is None,
              f"top cos={out_m['matches'][0]['cos'] if out_m['matches'] else 0}, "
              f"2-й вид cos={out_m['matches'][1]['cos'] if len(out_m['matches'])>1 else 0}")
        # но с threshold=0 — маржа мала -> всё равно отказ
        out_m2 = sys_.recognize(os.path.join(spn2, "a.jpg"), threshold=0.0)
        check("margin: отказ независим от порога при малом отрыве",
              not out_m2["certain"])

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\nИтог: {len(results) - n_fail}/{len(results)} тестов пройдено.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
