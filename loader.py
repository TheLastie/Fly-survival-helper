"""Загрузка длинных текстов: нарезка на эпизоды с перекрытием.

Мозг хранит эпизоды фиксированной «зернистости»: целый учебник целиком
дал бы один размытый вектор (деталь растворилась бы в среднем по тексту).
Поэтому длинный текст режется на скользящие окна предложений, каждое окно —
отдельный эпизод. Перекрытие (overlap) даёт устойчивость: деталь,
упомянутая на стыке двух окон, находится из обоих.
"""
import re

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)")


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text: str, size: int = 2, overlap: int = 1) -> list:
    """Скользящее окно из `size` предложений с шагом size - overlap."""
    sents = split_sentences(text)
    if not sents:
        return []
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(sents), step):
        ch = " ".join(sents[i:i + size]).strip()
        if ch:
            chunks.append(ch)
        if i + size >= len(sents):
            break
    return chunks


def chunk_doc(text: str, size: int = 2, overlap: int = 1) -> list:
    """Нарезка документа на (текст, заголовок) с учётом структуры md.

    Заголовок (# ...) открывает секцию и подставляется в meta эпизодов —
    поиск «внутри раздела» работает через фильтр по meta['heading'].
    """
    heading, buf, out = None, [], []

    def flush():
        nonlocal buf
        if buf:
            body = " ".join(buf)
            for ch in make_chunks(body, size, overlap):
                out.append((ch, heading))
            buf = []

    for ln in text.splitlines():
        m = _HEADING_RE.match(ln.strip())
        if m:
            flush()
            heading = m.group(1).strip()
            continue
        buf.append(ln)
    flush()
    if out:
        return out
    return [(t, None) for t in make_chunks(text, size, overlap)]


def load_text(system, text: str, source: str | None = None,
              size: int = 2, overlap: int = 1) -> list:
    """Нарезать текст и one-shot записать каждый фрагмент. Возвращает индексы."""
    ids = []
    for ch in make_chunks(text, size, overlap):
        x = system.embedder.embed(ch)
        idx = system.memory.store(ch, x, meta={"source": source} if source else None)
        system.brain.learn(x, reward=+1)
        ids.append(idx)
    return ids
