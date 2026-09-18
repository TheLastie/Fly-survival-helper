"""Импорт World Flora Online Plant List -> каркас базы видов.

Потоковая двухпроходная обработка (без загрузки ~1 ГБ TSV в RAM):
  проход 1 (name.tsv):   индекс имен по рангам species/genus/family;
  проход 2 (taxon.tsv):  для таксонов этих имён собираем parentID;
  сборка: вид -> род -> семейство.

Выходы:
  --out backbone.tsv:  wfo_id, name, genus, family (принятые виды);
  --skeleton-dir:      виды/<Род>_specificEpithet/card.md — заготовки
                       карточек для выбранного семейства (--family).

Запуск:
  python wfo_import.py --zip wfo_plantlist_2026-06.zip \
      --out flora_backbone.tsv --family Pinaceae --skeleton-dir виды/
"""
import argparse
import csv
import io
import sys
import zipfile

SPECIES_RANKS = ("species",)
GENUS_RANKS = ("genus",)
FAMILY_RANKS = ("family",)


def pass_names(zf: zipfile.ZipFile) -> dict:
    """name.tsv -> {nameID: (scientificName, genus, rank)} для нужных рангов."""
    names = {}
    with zf.open("name.tsv") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        for row in csv.DictReader(text, delimiter="\t"):
            rank = row.get("rank", "")
            if rank in SPECIES_RANKS + GENUS_RANKS + FAMILY_RANKS:
                names[row["ID"]] = (row.get("scientificName", "").strip(),
                                    row.get("genus", "").strip(), rank)
    return names


def pass_taxa(zf: zipfile.ZipFile, names: dict) -> dict:
    """taxon.tsv -> {taxonID: (name, genus, rank, parentID)} для известных имён."""
    taxa = {}
    with zf.open("taxon.tsv") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        for row in csv.DictReader(text, delimiter="\t"):
            nid = row.get("nameID", "")
            if nid in names:
                nm, gen, rank = names[nid]
                taxa[row["ID"]] = (nm, gen, rank, row.get("parentID", ""))
    return taxa


def build_backbone(taxa: dict) -> list:
    """вид -> (имя, род, семейство). Род/семейство — через parent-цепочку."""
    by_name_rank = {}
    for tid, (nm, gen, rank, parent) in taxa.items():
        by_name_rank.setdefault((nm, rank), tid)
    backbone = []
    for tid, (nm, gen, rank, parent) in taxa.items():
        if rank != "species":
            continue
        genus_name, family_name = gen, ""
        g = taxa.get(parent)
        if g:
            genus_name = g[0] or genus_name
            fam = taxa.get(g[3])
            if fam:
                family_name = fam[0]
        backbone.append((tid, nm, genus_name, family_name))
    return backbone


def emit_skeletons(backbone: list, family: str, out_dir: str, limit: int):
    import os
    n = 0
    for tid, nm, gen, fam in backbone:
        if fam != family or n >= limit:
            continue
        parts = nm.split()
        if len(parts) < 2:
            continue
        dirname = f"{parts[0]}_{parts[1]}"
        d = os.path.join(out_dir, dirname)
        os.makedirs(d, exist_ok=True)
        card = os.path.join(d, "card.md")
        if not os.path.exists(card):
            with open(card, "w", encoding="utf-8") as f:
                f.write(f"# {nm}\n\n"
                        f"- семейство: {fam}\n"
                        f"- род: {gen}\n"
                        f"- wfo: {tid}\n"
                        f"- ядовит: неизвестно\n\n"
                        f"> Заполнить: описание, съедобность, похожие виды,\n"
                        f"> применение в выживании. Фото положить рядом (*.jpg).\n")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="wfo_plantlist_2026-06.zip")
    ap.add_argument("--out", default="flora_backbone.tsv")
    ap.add_argument("--family", default=None, help="фильтр для скелетов")
    ap.add_argument("--skeleton-dir", default=None)
    ap.add_argument("--skeleton-limit", type=int, default=500)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    print("проход 1: имена...", flush=True)
    names = pass_names(zf)
    print(f"  имена нужных рангов: {len(names):,}", flush=True)
    print("проход 2: таксоны...", flush=True)
    taxa = pass_taxa(zf, names)
    print(f"  таксоны: {len(taxa):,}", flush=True)
    print("сборка backbone...", flush=True)
    backbone = build_backbone(taxa)
    print(f"  принятых видов: {len(backbone):,}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("wfo_id\tname\tgenus\tfamily\n")
        for row in backbone:
            f.write("\t".join(row) + "\n")
    print(f"backbone -> {args.out}")

    if args.family and args.skeleton_dir:
        n = emit_skeletons(backbone, args.family, args.skeleton_dir,
                           args.skeleton_limit)
        print(f"скелетов карточек ({args.family}) -> {args.skeleton_dir}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
