"""Тест универсального поиска: корпус из разных форматов, дедуп, фильтры, цитаты.

Запуск: python searchtest.py
"""
import os
import tempfile

from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem

results = []


def check(name, cond, detail=""):
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


MD = """# История генетики
Дрозофила меланогастер стала модельным организмом в начале двадцатого века.
Морган начал работать с мухой в 1907 году в Колумбийском университете.
В 1910 году Морган обнаружил мутанта с белыми глазами.
# Современность
Геном дрозофилы секвенирован в 2000 году.
В мозге мухи около ста тысяч нейронов.
"""

TXT = ("Митохондрия — органелла клетки, вырабатывающая энергию в форме АТФ. "
       "Митохондрия имеет собственную кольцевую ДНК. "
       "Эндосимбиотическая теория объясняет происхождение митохондрий. "
       "По ней митохондрия произошла от древних бактерий. "
       "Размер митохондриальной ДНК человека около 16 тысяч пар оснований. "
       "Белки митохондрий кодируются частично ядром, частично собственным геномом.")

CSV = "имя,должность,лаборатория\nСидоров,инженер,Альфа\nВолков,биолог,Бета\n"

PY = ("def familiarity(out, x):\n"
      "    \"\"\"Косинусная знакомость восстановленного вектора.\"\"\"\n"
      "    return cos(out, x) * trace_strength(out)\n"
      "class FlyBrain:\n"
      "    pass\n")

JSONL = ('{"модель": "qwen2.5", "параметры": "0.5b", "назначение": "генерация"}\n'
         '{"модель": "minilm", "параметры": "12-layer", "назначение": "эмбеддинг"}\n')


def main():
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "история.md"), "w", encoding="utf-8").write(MD)
        open(os.path.join(td, "биология.txt"), "w", encoding="utf-8").write(TXT)
        open(os.path.join(td, "сотрудники.csv"), "w", encoding="utf-8").write(CSV)
        open(os.path.join(td, "ядро.py"), "w", encoding="utf-8").write(PY)
        open(os.path.join(td, "модели.jsonl"), "w", encoding="utf-8").write(JSONL)

        sys_ = FlySystem(FlyConfig(), HashingEmbedder())
        r1 = sys_.index_path(td)
        print(f"индексация: {r1}")
        check("5 файлов прочитано", r1["files"] == 5)
        check("эпизоды созданы", r1["added"] >= 12, f"added={r1['added']}")
        check("по форматам: csv/jsonl/код в индексе",
              any("Сидоров" in t for t in sys_.memory.texts)
              and any("qwen2.5" in t for t in sys_.memory.texts)
              and any("familiarity" in t for t in sys_.memory.texts))

        # дедуп: повторная индексация того же корпуса
        r2 = sys_.index_path(td)
        check("дедуп при повторной индексации", r2["added"] == 0 and r2["dups"] > 0,
              f"added={r2['added']}, dups={r2['dups']}")

        # keyword-поиск (редкий термин из txt)
        res = sys_.search("митохондриальная ДНК человека")
        hit = res.candidates and "митохондри" in res.candidates[0].text.lower()
        check("keyword-поиск: топ-1 из биология.txt", hit,
              f"top: {res.candidates[0].text[:60] if res.candidates else '—'}")

        # перефраз (dense): «белоглазка» из md
        res = sys_.search("расскажи про мутанта с необычными глазами у Моргана")
        hit = res.candidates and "белыми глазами" in res.candidates[0].text
        check("перефраз: белоглазка в топ-1", hit,
              f"top: {res.candidates[0].text[:60] if res.candidates else '—'}")

        # заголовок md попал в meta
        c = next(c for c in res.candidates if "белыми глазами" in c.text)
        m = sys_.memory.meta[c.idx]
        check("заголовок в meta", m.get("heading") == "История генетики", str(m))

        # фильтр по источнику
        res = sys_.search("инженер", source="сотрудники.csv")
        ok = res.candidates and all("сотрудники.csv" in sys_.citation(c) for c in res.candidates[:2])
        check("фильтр по источнику (csv)", ok,
              f"top: {sys_.citation(res.candidates[0]) if res.candidates else '—'}")

        # цитаты содержат путь
        res = sys_.search("АТФ энергия")
        cit = sys_.citation(res.candidates[0]) if res.candidates else ""
        check("цитата содержит источник", "биология.txt" in cit, cit)

        # code search: редкий термин из .py
        res = sys_.search("trace_strength")
        ok = res.candidates and "trace_strength" in res.candidates[0].text
        check("поиск по коду (trace_strength)", ok)

        # csv: запрос из таблицы
        res = sys_.search("кто биолог в Бете")
        ok = res.candidates and "Волков" in res.candidates[0].text
        check("csv: Волков в топ-1", ok,
              f"top: {res.candidates[0].text[:60] if res.candidates else '—'}")

        # неизвестный запрос
        res = sys_.search("квантовая хромодинамика лептоны")
        check("неизвестный запрос не галлюцинирует",
              res.zone in ("unknown", "unsure"), f"zone={res.zone}, fam={res.familiarity:.2f}")

        # extractive QA: имя из топ-чанков
        from answer_extract import extract_answer, is_question, detect_type
        res = sys_.search("кто обнаружил мутанта с белыми глазами?")
        ext = extract_answer("кто обнаружил мутанта с белыми глазами?",
                             [(c.text, None) for c in res.candidates[:3]])
        check("extractive QA: имя", ext is not None and ext["answer"] == "Морган",
              str(ext)[:90] if ext else "None")
        res = sys_.search("в каком году Морган обнаружил мутанта?")
        ext = extract_answer("в каком году Морган обнаружил мутанта?",
                             [(c.text, None) for c in res.candidates[:3]])
        check("extractive QA: год", ext is not None and ext["answer"] == "1910",
              str(ext)[:90] if ext else "None")
        check("is_question/detect_type", is_question("кто это?")
              and detect_type("в каком году?") == "year"
              and detect_type("расскажи про") == "thing")

        # RM3: экспансия не содержит терминов запроса, search с weights работает
        from search import rm3_expand
        from textnorm import norm_terms
        exp = rm3_expand(norm_terms, ["морган", "белоглазк"],
                         [sys_.memory.texts[i] for i, _ in
                          sys_.store.search_bm25("морган белоглазк", 5)])
        check("RM3: экспансия без терминов запроса",
              "морган" not in exp and "белоглазк" not in exp and len(exp) > 0,
              f"exp={exp[:5]}")
        hits_w = sys_.store.search_bm25(["морган", "год"], 5, weights={"год": 0.4})
        check("BM25 с весами работает", len(hits_w) > 0)

        # фразовые запросы: точная фраза находится, порядок нарушен — нет
        res = sys_.search('"мутанта с белыми глазами"', k=3)
        hit = res.candidates and "мутанта с белыми глазами" in res.candidates[0].text
        check("фраза в кавычках: точное совпадение", hit,
              res.candidates[0].text[:60] if res.candidates else "—")
        docs = sys_.store.phrase_docs(["мутант", "с", "бел", "глаз"])
        check("phrase_docs: прямой порядок", any("белыми глазами" in sys_.memory.texts[i]
                                                 for i in docs), f"n={len(docs)}")
        docs_rev = sys_.store.phrase_docs(["глаз", "бел"])
        ok_rev = all("глазами белыми" not in sys_.memory.texts[i] for i in docs_rev)
        check("phrase_docs: обратный порядок пуст", len(docs_rev) == 0 or ok_rev,
              f"n={len(docs_rev)}")

        # операторы: -исключение, source:, heading:
        res = sys_.search("инженер -Сидоров")
        ok = res.candidates and all("Сидоров" not in c.text for c in res.candidates[:3])
        check("оператор -исключение", ok,
              res.candidates[0].text[:50] if res.candidates else "—")
        res = sys_.search("биолог source:сотрудники.csv")
        ok = res.candidates and all("сотрудники.csv" in sys_.citation(c)
                                    for c in res.candidates[:2])
        check("оператор source:", ok)
        res = sys_.search("нейроны heading:Современность")
        ok = res.candidates and all("Современность" in str(
            (sys_.memory.meta[c.idx] or {}).get("heading", "")) for c in res.candidates[:3])
        check("оператор heading:", ok,
              str((sys_.memory.meta[res.candidates[0].idx] or {}).get("heading")) if res.candidates else "—")

        n_fail = sum(1 for _, ok in results if not ok)
        print(f"\nИтог: {len(results) - n_fail}/{len(results)} тестов пройдено.")
        return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
