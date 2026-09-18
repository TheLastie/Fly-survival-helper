"""Языковое ядро (этап E2): стеммер RU/EN, раскладка, symspell-инструменты.

Принципы:
  — ноль внешних словарей: частоты — df из самого индекса;
  — исправление опечаток — только токены запроса, отсутствующие в словаре,
    длина 4..28;
  — раскладка — только в пути запроса (индекс хранит оригинал);
  — стеммер суффиксный, консервативный: лучше недостеммировать, чем
    склеить разные слова (эмбеддинг-слой перекрывает недостатки).
"""
import re

from memory import tokenize as _tokenize

CYR = set("абвгдежзиклмнопрстуфхцчшщъыьэю")
LAT = set("abcdefghijklmnopqrstuvwxyz")

# ---------------------------------------------------------------- раскладка
_QW2RU = str.maketrans(
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./`",
    "йцукенгшщзхъфывапролджэячсмитьбю.ё")
_RU2QW = str.maketrans(
    "йцукенгшщзхъфывапролджэячсмитьбю.ё",
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./`")


def fix_layout_token(tok: str, vocab: set | None) -> str:
    """Если токен (в стеммированном виде) неизвестен, а его перекладка
    известна — вернуть перекладку. Иначе исходный токен."""
    if vocab is None:
        return tok
    if stem(tok) in vocab:
        return tok
    conv = tok.translate(_QW2RU)
    if conv != tok and stem(conv) in vocab:
        return conv
    conv = tok.translate(_RU2QW)
    if conv != tok and stem(conv) in vocab:
        return conv
    return tok


# ------------------------------------------------------------------ стеммер
# Порядок в списке = порядок применения; выигрывает самый длинный суффикс.
# После отрезки основа должна быть >= 3 символов (иначе откат).
_RU_SFX = [
    # превосходная степень
    "ейшего", "ейшему", "ейшими", "ейших", "ейшее", "ейший",
    # причастия (-ущ/-ющ/-вш/-нн), длинные формы первыми
    "ющимися", "ющемся", "ющегося",
    "ющего", "ющему", "ющими", "ующих", "ющих",
    "ующая", "ющая", "ующее", "ющее", "ующие", "ющие",
    "ующим", "ющим", "ующий", "ющий", "ующую", "ющую", "ющим", "ющей",
    "вшегося", "вшемуся", "вшимися", "вшихся", "вшееся", "вшийся",
    "вшего", "вшему", "вшими", "вших", "вшая", "вшее", "вшие",
    "вший", "вшим", "вшую", "вше",
    "ннего", "ннему", "нными", "нных", "нная", "нное", "нные",
    "нный", "нным", "нную",
    # глагольные формы
    "ывающ", "ивающ", "ывают", "ивают", "ывает", "ивает", "ывая", "ивая",
    "ывали", "ивали", "овали", "евали", "овало", "евало", "овала", "евала",
    "оваться", "овать", "евать", "иваясь", "ываясь", "ивать", "ывать",
    "аете", "яете", "аешь", "яешь",
    "ает", "яет", "ают", "яют", "али", "яли", "ало", "яло",
    "ать", "ять", "оть", "уть", "ыть", "ить", "еть",
    "им", "ишь", "ите", "ил", "ила", "ило", "или",
    "ут", "ют", "ат", "ит", "ят", "ем", "ешь",
    "енн", "ен", "ан", "ян", "он", "ин", "т",
    # существительные и короткие падежи
    "иями", "иях", "иям", "ией", "ием",
    "аями", "ями", "ыми", "ими", "ами", "ах", "ям", "ов", "ев", "ей", "ам",
    "ого", "ему", "ому", "их", "ых",
    "ая", "ое", "ые", "ий", "ый", "ой", "ую", "юю", "ее", "ие",
    "ем", "ом", "ью", "ия", "ие", "ий",
    "а", "о", "у", "е", "ы", "и", "ь", "я", "ю",
]

_EN_VOWELS = set("aeiouy")


def _stem_ru(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ся") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("сь") and len(w) > 4:
        w = w[:-2]
    for s in _RU_SFX:
        if w.endswith(s):
            base = w[:-len(s)]
            if len(base) >= 3:
                return base
            return w
    return w


def _stem_en(w: str) -> str:
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith("'s"):
        w = w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5 and any(c in _EN_VOWELS for c in w[:-3]):
        w = w[:-3]
        if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in _EN_VOWELS:
            w = w[:-1]
        return w
    if w.endswith("ed") and len(w) > 4 and any(c in _EN_VOWELS for c in w[:-2]):
        return w[:-2]
    if w.endswith("oes") and len(w) > 4:
        return w[:-3] + "o"
    if w.endswith("es") and w.endswith(("ses", "xes", "ches", "shes", "zes")):
        return w[:-2]
    if w.endswith("ly") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def stem(tok: str) -> str:
    """Стемминг по алфавиту токена: кириллица -> RU, латиница -> EN."""
    if not tok:
        return tok
    c0 = tok[0]
    if c0 in CYR or c0 == "ё":
        return _stem_ru(tok.replace("ё", "е"))
    if c0 in LAT:
        return _stem_en(tok)
    return tok


def norm_terms(text: str) -> list:
    """Нормализованные термины для ИНДЕКСАЦИИ (без раскладки/spell)."""
    return [stem(t) for t in _tokenize(text)]


# --------------------------------------------------------------- symspell
def spell_deletes(term: str, max_edit: int = 2) -> set:
    """Все строки, получаемые удалением 1..max_edit символов."""
    out, prev = set(), {term}
    for _ in range(max_edit):
        cur = set()
        for w in prev:
            for i in range(len(w)):
                cur.add(w[:i] + w[i + 1:])
        out |= cur
        prev = cur
    return out


def edit_dist(a: str, b: str, cutoff: int = 3) -> int:
    """Оптимизированное расстояние Дамерау-Левенштейна с ранним выходом."""
    la, lb = len(a), len(b)
    if abs(la - lb) >= cutoff:
        return cutoff
    if la == 0 or lb == 0:
        return max(la, lb)
    prev2 = None
    prev1 = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            v = min(prev1[j] + 1, cur[j - 1] + 1, prev1[j - 1] + cost)
            if (i > 1 and j > 1 and prev2 is not None
                    and ca == b[j - 2] and a[i - 2] == b[j - 1]):
                v = min(v, prev2[j - 2] + cost)
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min >= cutoff:
            return cutoff
        prev2, prev1 = prev1, cur
    return prev1[lb]


if __name__ == "__main__":
    cases = [
        ("арбалета", "арбалет"), ("белоглазки", "белоглазк"),
        ("работающими", "работа"), ("белыми", "бел"), ("синими", "син"), ("дрозофилы", "дрозофил"),
        ("грибовидное", "грибовидн"), ("Трудно", "трудн"),
        ("walking", "walk"), ("heroes", "hero"), ("studies", "study"),
    ]
    bad = [(a, stem(a), e) for a, e in cases if stem(a.lower()) != e]
    assert not bad, bad
    assert edit_dist("арбалет", "орболет") == 2
    assert edit_dist("вихрь", "вихрь") == 0
    assert edit_dist("книга", "книгв") == 1
    assert "вхр" in spell_deletes("вихрь")
    print("textnorm: все проверки пройдены")
