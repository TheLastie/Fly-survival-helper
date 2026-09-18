"""Extractive QA без генеративной модели (P5 плана).

Для вопросных запросов достаёт сущность нужного типа из топ-чанков:
  name/place — самая частая именная группа заданного вида, отсутствующая
               в самом запросе (классический приём: ответ — то, о чём
               спрашивают, значит в запросе его нет);
  year/number — то же для чисел;
  reason/thing — лучшее предложение-обоснование (макс. пересечение
               терминов с запросом, взвешенное рангом чанка).
"""
import re
from collections import Counter

from textnorm import norm_terms

NAME_RE = re.compile(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)*")
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUM_RE = re.compile(r"\b\d[\d\s]*(?:[.,]\d+)?\b")
PREP_RE = re.compile(r"\b(?:в|во|на|из|с|со|у|к|от|до)\s+([А-ЯЁ][а-яё]+)")

QUESTION_WORDS = ("как", "кто", "что", "где", "когда", "почему", "зачем",
                  "сколько", "который", "которая", "чей", "откуда", "куда",
                  "какой", "какая", "какие", "какое", "кем", "чем")

STOPNAMES = {"он", "она", "они", "его", "ее", "их", "это", "та", "тот",
             "все", "всё", "она", "этот", "эта", "эти", "свой", "своя",
             # служебные/диалоговые слова, ловимые как «имена» с заглавной
             "есть", "как", "что", "да", "нет", "ну", "вот", "так", "уже",
             "еще", "ещё", "очень", "теперь", "здесь", "там", "сейчас",
             "просто", "конечно", "хорошо", "ладно", "может", "нужно",
             "можно", "будто", "правда", "впрочем", "один", "одна"}

# типичные слова-стартеры предложений (не имена собственные)
STARTERS = {"эксперименты", "однако", "вскоре", "потом", "затем", "так",
            "там", "здесь", "тогда", "словом", "итак", "вместе",
            "наконец", "после", "перед", "вечером", "утром", "ночью"}

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def is_question(q: str) -> bool:
    q = q.strip().lower()
    return q.endswith("?") or q.split()[0] in QUESTION_WORDS if q else False


def detect_type(q: str) -> str | None:
    ql = q.lower()
    if re.search(r"в каком году|в каком веке|когда\b", ql):
        return "year"
    if re.search(r"сколько\b", ql):
        return "number"
    if re.search(r"как (звал|звали|назыв|зовут)|кто\b|кем\b|чей\b", ql):
        return "name"
    if re.search(r"где\b|куда\b|откуда\b", ql):
        return "place"
    if re.search(r"почему|зачем|отчего", ql):
        return "reason"
    if re.search(r"что\b|чем\b|расскажи|перескажи|опиши|объясни|поведай", ql):
        return "thing"
    return None


def _sentences(text: str) -> list:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 5]


def extract_answer(query: str, chunks: list, topn: int = 3):
    """chunks: [(text, meta), ...] по убыванию релевантности.
    Возвращает dict(answer, evidence, conf, type) или None."""
    qt = set(norm_terms(query))
    qtext = query.lower()
    atype = detect_type(query)
    # вес предложения: ранг чанка x пересечение терминов с запросом.
    # Сущность должна добываться из предложения, релевантного запросу,
    # а не просто из топ-чанка (иначе соседние факты перебивают ответ).
    # идентичные предложения (дубли оверлейных окон чанковой нарезки)
    # учитываем один раз, по максимальному весу
    best_by_sent: dict[str, tuple[float, int]] = {}
    for rank, (text, _meta) in enumerate(chunks[:topn]):
        rank_w = 1.0 / (rank + 1)
        for s in _sentences(text):
            ov = len(set(norm_terms(s)) & qt)
            if ov == 0:
                continue  # предложение не про этот запрос
            w = rank_w * ov * (1.0 + 0.1 * ov)
            if s not in best_by_sent or w > best_by_sent[s][0]:
                best_by_sent[s] = (w, rank)
    weighted = [(w, s, rank) for s, (w, rank) in best_by_sent.items()]
    if not weighted:
        return None

    def entities(rx, filt=None):
        c = Counter()
        chunks_seen: dict[str, set] = {}
        for w, s, rank in weighted:
            for m in rx.finditer(s):
                val = m.group(1) if m.groups() else m.group(0)
                if len(val) <= 2 or val.lower() in qtext:
                    continue
                if m.start() == 0 and val.lower() in STARTERS:
                    continue  # заглавная из-за начала предложения
                if filt and not filt(val):
                    continue
                c[val] += w
                chunks_seen.setdefault(val, set()).add(rank)
        # буст только за повтор в НЕ-соседних чанках: соседние ранги —
        # это дубли оверлейной нарезки, а не истинный повтор сущности
        for val, ranks in chunks_seen.items():
            if max(ranks) - min(ranks) >= 2:
                c[val] *= 1.5
        return c

    if atype == "name":
        c = entities(NAME_RE, lambda v: v.lower() not in STOPNAMES)
    elif atype == "place":
        c = entities(PREP_RE)
        if not c:
            c = entities(NAME_RE, lambda v: v.lower() not in STOPNAMES)
    elif atype == "year":
        c = entities(YEAR_RE)
    elif atype == "number":
        c = entities(NUM_RE, lambda v: v.strip() not in qtext)
    else:
        c = Counter()

    if c:
        answer, cnt = c.most_common(1)[0]
        evidence = next((s for w, s, _r in weighted if answer in s), "")
        conf = min(0.95, 0.45 + 0.18 * cnt)
        return {"answer": answer, "evidence": evidence, "conf": round(conf, 2),
                "type": atype}

    # reason/thing/fallback: лучшее предложение (вес уже включает пересечение)
    best_s, best_sc = None, 0.0
    for w, s, _rank in weighted:
        if w > best_sc:
            best_s, best_sc = s, w
    if best_s and best_sc > 1.0:
        return {"answer": best_s, "evidence": best_s, "conf": 0.5, "type": atype}
    return None


if __name__ == "__main__":
    chunks = [
        ("В 1910 году Морган обнаружил мутанта с белыми глазами. "
         "Эксперименты с белоглазкой привели к открытию сцепления генов с полом.", {}),
        ("Морган сформулировал хромосомную теорию наследственности в 1915 году. "
         "Вместе с ним работали студенты: Стёртевант, Мёллер и Бриджес.", {}),
    ]
    r = extract_answer("кто обнаружил мутанта с белыми глазами?", chunks)
    assert r and r["answer"] == "Морган" and r["type"] == "name", r
    r = extract_answer("в каком году Морган обнаружил мутанта?", chunks)
    assert r and r["answer"] == "1910" and r["type"] == "year", r
    r = extract_answer("с кем работал Морган?", chunks)
    assert r and "Стёртевант" in r["answer"], r
    r = extract_answer("перескажи про хромосомную теорию", chunks)
    assert r and r["type"] == "thing", r
    print("answer_extract: все проверки пройдены")
