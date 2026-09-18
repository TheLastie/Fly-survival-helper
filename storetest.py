"""Тесты сегментного индекса (E1): CSR-BM25 против brute-force, roundtrip,
tombstones, скорость переоткрытия.

Запуск: python storetest.py [--book <путь>]  (с --book добавляется тест
на реальном корпусе: переоткрытие ~20k документов должно быть < 5 с)
"""
import argparse
import os
import tempfile
import time

import numpy as np

from config import FlyConfig
from embedder import HashingEmbedder
from index_store import BM25_B, BM25_K1, IndexStore
from memory import tokenize
from textnorm import norm_terms
from system import FlySystem

results = []


def check(name, cond, detail=""):
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def brute_bm25(store, terms, docid):
    """Эталонный BM25 одного документа напрямую по тексту."""
    from collections import Counter
    c = Counter(norm_terms(store.texts[docid]))
    dl = sum(c.values())
    n = len(store.texts)
    s = 0.0
    for t in set(norm_terms(terms)):
        tf = c.get(t, 0)
        if tf == 0:
            continue
        df = sum(1 for x in store.texts if t in norm_terms(x))
        idf = float(np.log1p((n - df + 0.5) / (df + 0.5)))
        s += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / store.avgdl))
    return s


def corpus(n=400, seed=0):
    rng = np.random.default_rng(seed)
    words = ["архипелаг", "баран", "вихрь", "гроза", "дубина", "ерш", "жаба",
             "зонт", "иней", "колос", "ломоть", "маятник", "носорог", "откос"]
    texts = []
    for i in range(n):
        k = int(rng.integers(8, 25))
        t = " ".join(str(rng.choice(words)) for _ in range(k))
        texts.append(f"документ {i}: {t}")
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default=None)
    args = ap.parse_args()

    texts = corpus()
    metas = [{"source": f"/corpus/a.txt", "pos": i} if i % 2 else
             {"source": f"/corpus/b.txt", "pos": i} for i in range(len(texts))]
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(len(texts), 64)).astype(np.float32)

    store = IndexStore()
    store.add_docs(texts, metas, vecs)
    store.finalize()
    check("CSR построен", len(store.terms) > 10 and store.ptr is not None,
          f"terms={len(store.terms)}, postings={len(store.pdocs)}")

    q = "архипелаг вихрь жаба"
    hits = store.search_bm25(q, 5)
    ok = True
    for gid, score in hits[:3]:
        ref = brute_bm25(store, q, gid)
        ok &= abs(score - ref) < 1e-4
    check("BM25 CSR == brute-force", ok, f"hits={len(hits)}")

    # roundtrip
    with tempfile.TemporaryDirectory() as td:
        store.commit(td)
        t0 = time.time()
        s2 = IndexStore.open(td)
        dt = time.time() - t0
        check("roundtrip: тексты и векторы", s2.texts == store.texts
              and np.allclose(s2.vecs.astype(np.float32), store.vecs.astype(np.float32)))
        h2 = s2.search_bm25(q, 5)
        check("roundtrip: те же топ-5 BM25",
              [i for i, _ in h2] == [i for i, _ in hits])

        # инкремент: второй коммит добавляет сегмент
        store.add_docs(["новый документ про зонт и ерша"],
                       [{"source": "/corpus/c.txt"}], np.ones((1, 64), np.float32))
        store.finalize()
        store.commit(td)
        s3 = IndexStore.open(td)
        check("инкремент: 2 сегмента, документ виден",
              len(s3.texts) == len(store.texts) and "зонт" in s3.texts[-1],
              f"segs={len(os.listdir(td)) - 1}")

        # tombstones
        store.remove_source("/corpus/a.txt")
        store.commit(td)
        s4 = IndexStore.open(td)
        hits_a = [i for i, _ in s4.search_bm25(["документ"], 500)]
        check("tombstone: source 'a' исключён",
              all("/corpus/a.txt" not in str(s4.metas[i].get("source", ""))
                  for i in hits_a),
          f"deleted={int(s4.deleted.sum())}")

    # системный уровень: save/load через FlySystem
    with tempfile.TemporaryDirectory() as td:
        sys_ = FlySystem(FlyConfig(), HashingEmbedder())
        sys_.index_path  # noqa — просто проверка атрибута
        for i in range(20):
            sys_.memorize(f"факт номер {i} про мух и грибовидные тела")
        sys_.save(td)
        sys2 = FlySystem(FlyConfig(), HashingEmbedder())
        sys2.load(td)
        fam_a = sys_.brain.familiarity(sys_.embedder.embed("факты про мух"))
        fam_b = sys2.brain.familiarity(sys2.embedder.embed("факты про мух"))
        check("system save/load (новый формат)", abs(fam_a - fam_b) < 1e-4,
              f"fam {fam_a:.3f} vs {fam_b:.3f}")
        # повторный save в ту же папку — снапшот, без дублирования сегментов
        sys_.save(td)
        sys3 = FlySystem(FlyConfig(), HashingEmbedder())
        sys3.load(td)
        check("повторный save: нет дубликатов", len(sys3.memory) == len(sys_.memory),
              f"{len(sys3.memory)} vs {len(sys_.memory)}")
        # re-embed: memorize до fit -> вектор факта переезжает в SIF-пространство
        import numpy as _np
        sys5 = FlySystem(FlyConfig(), HashingEmbedder())
        sys5.memorize("Морган открыл наследование белоглазки у дрозофилы")
        v0 = sys5.memory.vecs[0].copy()
        with tempfile.TemporaryDirectory() as td2:
            open(os.path.join(td2, "big.txt"), "w", encoding="utf-8").write(
                "\n".join(f"служебный документ номер {i} про разное." for i in range(60)))
            sys5.index_path(td2)
        check("re-embed после fit: вектор обновлён",
              not _np.allclose(v0, sys5.memory.vecs[0]),
              "vecs differ")
        check("re-embed: store и memory синхронны",
              _np.allclose(sys5.store.vecs[0].astype(_np.float32),
                           sys5.memory.vecs[0], atol=1e-3))
        x = sys5.embedder.embed("наследование белоглазки")
        check("re-embed: факт находится dense top-1",
              sys5.memory.topk(x, 1)[0][0] == 0)

        # консолидация: системно наказываемый эпизод деактивируется
        sys6 = FlySystem(FlyConfig(), HashingEmbedder())
        sys6.memorize("ядовитый мох никогда не ешь")
        r6 = sys6.search("мох", k=3)
        c6 = next((c for c in r6.candidates if "мох" in c.text), None)
        if c6 is not None:
            sys6.feedback(-1, cand=c6)
            sys6.feedback(-1, cand=c6)
            rep = sys6.consolidate()
            check("консолидация: наказываемый деактивирован",
                  rep["deactivated"] >= 1 and any(not a for a in sys6.memory.active),
                  str(rep))

        # инвариант docid: memory index == store docid (ловит рассинхроны
        # путей добавления — баг index_species v1)
        check("инвариант docid memory==store",
              all(sys_.memory.texts[i] == sys_.store.texts[i]
                  for i in range(len(sys_.store))),
              f"n={len(sys_.store)}")

        # сессионный контекст: «он» разрешается через предыдущий запрос
        with tempfile.TemporaryDirectory() as td3:
            open(os.path.join(td3, "герои.txt"), "w", encoding="utf-8").write(
                "Румата жил в Арканаре. Арканар был столицей государства. "
                "Кира жила в доме на окраине Арканара. "
                "Петров жил в Москве. Москва была большим городом на севере.")
            def _fresh():
                fs = FlySystem(FlyConfig(), HashingEmbedder())
                fs.index_path(td3)
                return fs
            plain = _fresh().search("где он жил", k=3).candidates[0].text
            ctx_sys = _fresh()
            ctx_sys.search("Румата", k=3)
            ctx = ctx_sys.search("где он жил", k=3).candidates[0].text
            check("сессия: контекст разрешает «он»",
                  "Арканар" in ctx and "Арканар" not in plain,
                  f"ctx={ctx[:30]!r} plain={plain[:30]!r}")
            ctx_sys.new_topic()
            reset = ctx_sys.search("где он жил", k=3).candidates[0].text
            check("сессия: new_topic() сбрасывает", "Арканар" not in reset,
                  f"reset={reset[:30]!r}")

        # коррекции: +1 поднимает, -1 снимает, персистятся
        sys_.store.finalize()
        res = sys_.search("факт номер 3 про мух", k=3)
        if len(res.candidates) >= 2:
            tgt = res.candidates[1].idx
            sys_.feedback(+1, cand=res.candidates[1])
            res2 = sys_.search("факт номер 3 про мух", k=3)
            order = [c.idx for c in res2.candidates]
            pos = order.index(tgt) + 1 if tgt in order else 99
            check("коррекция: +1 поднимает на 1", pos == 1, f"pos={pos}")
            sys_.feedback(-1, cand=res2.candidates[0])
            res3 = sys_.search("факт номер 3 про мух", k=3)
            order3 = [c.idx for c in res3.candidates]
            pos3 = order3.index(tgt) + 1 if tgt in order3 else 99
            check("коррекция: -1 снимает", pos3 != 1, f"pos={pos3}")

    if args.book:
        sys_ = FlySystem(FlyConfig(decay=0.0), HashingEmbedder())
        t0 = time.time()
        r = sys_.index_path(args.book, size=3, overlap=1)
        t_index = time.time() - t0
        print(f"  книга: {r['added']} эпизодов, {t_index:.1f} с")
        with tempfile.TemporaryDirectory() as td:
            sys_.save(td)
            sz = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fns in os.walk(td) for f in fns)
            t0 = time.time()
            s2 = FlySystem(FlyConfig(), HashingEmbedder())
            s2.load(td)
            t_open = time.time() - t0
            check("книга: переоткрытие < 5 с", t_open < 5.0,
                  f"{t_open:.2f} с, {sz/1e6:.1f} МБ на диске")
            q = "как звали главного героя трудно быть богом"
            res = s2.search(q)
            check("книга: поиск после переоткрытия работает",
                  len(res.candidates) > 0, f"zone={res.zone}, top={res.candidates[0].text[:50] if res.candidates else '—'}")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\nИтог: {len(results) - n_fail}/{len(results)} тестов пройдено.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
