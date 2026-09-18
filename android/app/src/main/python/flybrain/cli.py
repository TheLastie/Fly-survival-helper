"""CLI-диалог с мозгом мухи (русский).

Команды:
  <любой текст>       — вопрос (узнавание)
  запомни: <текст>    — one-shot запоминание
  верно               — DAN +1: усилить последний ответ
  забудь [текст]      — DAN -1: стереть последний ответ / эпизод, содержащий текст
  что знаешь          — список активных воспоминаний
  сохрани <папка>     — сохранить состояние
  загрузи <папка>     — загрузить состояние
  выход               — завершить

Запуск:  python cli.py            (офлайн-эмбеддер)
         python cli.py --st      (sentence-transformers, если установлен)
"""
import argparse
import sys

from config import FlyConfig
from embedder import make_embedder
from system import FlySystem

BANNER = """\
=== Мозг мухи: ассоциативная память на грибовидном теле ===
Просто пиши вопрос или «запомни: <факт>». Ответы мозга содержат знакомость,
уверенность и запас. Оценивай ответы: «верно» / «забудь» — это дофамин (DAN)."""


def format_answer(res, cfg) -> str:
    fam = f"знакомость {res.familiarity:.2f}"
    if res.zone == "unknown":
        return (f"не помню ничего похожего ({fam} < {cfg.theta_low}).\n"
                f"  скажи «запомни: ...» — и я это выучу за один показ.")
    top = res.top
    conf = f"уверенность {top.conf:.0%}"
    if res.zone == "unsure":
        return f"кажется, речь о «{top.text}» ({fam}, {conf}) — но я не уверен. Так и есть?"
    if res.zone == "ambiguous":
        second = res.candidates[1]
        return (f"неоднозначно: похоже и на «{top.text}», и на «{second.text}» "
                f"(запас {res.margin:.3f}, {fam}). Уточни вопрос.")
    return f"{top.text}\n  [{fam}, {conf}, запас {res.margin:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", action="store_true", help="sentence-transformers вместо хэш-эмбеддера")
    ap.add_argument("--llama", default=None, metavar="HOST",
                    help="локальная LLM через llama-server (по умолчанию http://127.0.0.1:8081)")
    ap.add_argument("--ollama", default=None, metavar="MODEL",
                    help="LLM через локальный Ollama (модель вне проекта)")
    ap.add_argument("--load", default=None, help="загрузить состояние из папки при старте")
    args = ap.parse_args()

    cfg = FlyConfig()
    embedder = make_embedder("st" if args.st else "hash")
    cfg.d_model = embedder.d_model
    gen_backend = ("llama" if args.llama or args.llama is None and False else
                   "ollama" if args.ollama else "template")
    sys_ = FlySystem(cfg, embedder, gen_backend=gen_backend,
                     ollama_model=args.ollama or "qwen2.5:0.5b",
                     llama_host=args.llama or "http://127.0.0.1:8081")
    if args.load:
        sys_.load(args.load)

    print(BANNER)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nпока.")
            break
        if not q:
            continue
        low = q.lower()
        if low in ("выход", "exit", "quit"):
            print("пока.")
            break
        if low.startswith("запомни"):
            text = q.split(":", 1)[1].strip() if ":" in q else q[len("запомни"):].strip()
            if not text:
                print("  напиши: запомни: <факт>")
                continue
            st = sys_.memorize(text)
            print(f"  запомнил за один показ (знакомость стала {st['familiarity_after']:.2f}).")
        elif low.startswith("верно"):
            st = sys_.feedback(+1)
            if st:
                print(f"  DAN +1: ассоциация усилена (знакомость {st['familiarity_after']:.2f}).")
            else:
                print("  нечего усиливать — сначала спроси что-нибудь.")
        elif low.startswith("забудь"):
            text = q[len("забудь"):].strip() or None
            st = sys_.feedback(-1, text=text)
            if st:
                print(f"  DAN -1: след разрушен (знакомость {st['familiarity_after']:.2f}).")
            else:
                print("  не нашел, что стирать.")
        elif low in ("консолидация", "consolidate"):
            rep = sys_.consolidate()
            print(f"  освежено следов: {rep['refreshed']}, "
                  f"деактивировано: {rep['deactivated']}, "
                  f"PCA пересчитан: {'да' if rep['pca_refit'] else 'нет'}")
        elif low in ("новая тема", "сброс", "reset"):
            sys_.new_topic()
            print("  контекст сессии сброшен.")
        elif low in ("что знаешь", "что ты знаешь", "список"):
            facts = sys_.known_facts()
            if not facts:
                print("  пока ничего не помню.")
            for i, t in enumerate(facts, 1):
                print(f"  {i}. {t}")
        elif low.startswith("индексируй виды"):
            parts = q.split(None, 2)
            root = parts[2] if len(parts) > 2 else "."
            r = sys_.index_species(root)
            print(f"  виды: {r['species']} видов, {r['cards']} карточек, "
                  f"{r['images']} фото.")
        elif low.startswith("индексируй") or low.startswith("проиндексируй"):
            parts = q.split()
            root = parts[1] if len(parts) > 1 else "."
            size = int(parts[2]) if len(parts) > 2 else 2      # предложений в эпизоде
            overlap = int(parts[3]) if len(parts) > 3 else 1   # перекрытие окон
            r = sys_.index_path(root, size=size, overlap=overlap)
            print(f"  проиндексировано: {r['files']} файлов, "
                  f"{r['added']} эпизодов, {r['dups']} дублей пропущено.")
        elif low.startswith("найди"):
            query = q.split(None, 1)[1] if len(q.split(None, 1)) > 1 else ""
            if not query:
                print("  напиши: найди <запрос>")
                continue
            res = sys_.search(query)
            print(f"<  знакомость {res.familiarity:.2f} | зона {res.zone}")
            for i, c in enumerate(res.candidates[:5], 1):
                print(f"<  {i}. {c.text[:110]}")
                print(f"<     [{sys_.citation(c)} | conf {c.conf:.0%}]")
            if not res.candidates:
                print("<  ничего не нашёл.")
        elif low.startswith("распознай"):
            path = q.split(":", 1)[1].strip() if ":" in q else q[9:].strip()
            if not path:
                print("  напиши: распознай: <путь к фото>")
                continue
            out = sys_.recognize(path)
            if out["certain"]:
                print(f"<  {out['verdict']} (уверенность {out['matches'][0]['cos']:.2f})")
                print("<  " + (out.get("advice") or "").replace(chr(10), chr(10) + "<  "))
            else:
                print("<  НЕ УВЕРЕН — вердикта нет, не употребляй в пищу.")
                for m in out["matches"][:3]:
                    print(f"<    похоже на {m['species']} ({m['cos']:.2f})")
        elif low in ("источники", "что проиндексировано"):
            srcs = sys_.memory.sources()
            if not srcs:
                print("  индекс пуст.")
            for s, n in sorted(srcs.items()):
                print(f"  {s} — {n} эпизодов")
        elif low.startswith("сохрани"):
            path = q.split(None, 1)[1] if len(q.split(None, 1)) > 1 else "flybrain_state"
            sys_.save(path)
            print(f"  сохранено в {path}/")
        elif low.startswith("загрузи"):
            path = q.split(None, 1)[1] if len(q.split(None, 1)) > 1 else "flybrain_state"
            sys_.load(path)
            print(f"  загружено из {path}/ ({len(sys_.memory)} эпизодов)")
        else:
            res, text = sys_.answer(q)
            print("< " + text)
            if res.top is not None:
                print(f"<   [знакомость {res.familiarity:.2f} | уверенность {res.top.conf:.0%} "
                      f"| запас {res.margin:.3f} | зона {res.zone}]")
            else:
                print(f"<   [знакомость {res.familiarity:.2f} | зона {res.zone}]")


if __name__ == "__main__":
    main()
