"""Таблица памяти «вектор -> текст» и структуры результата recall.

Эмбеддинг обратимо не декодируется, поэтому тексты хранятся отдельно;
мозг (flybrain.py) отвечает за знакомость, таблица — за ранжирование кандидатов.

Универсальный поиск: кэшированная матрица векторов (быстрый dense top-k),
simhash-дедупликация при загрузке, метаданные источника в каждом эпизоде.
"""
import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower().replace("ё", "е"))


def simhash64(text: str) -> int:
    """64-битный simhash по токенам: для точного дедупа близких дублей."""
    v = np.zeros(64, dtype=np.float32)
    for t in tokenize(text):
        h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")
        for i in range(64):
            v[i] += 1.0 if (h >> i) & 1 else -1.0
    out = 0
    for i in range(64):
        if v[i] > 0:  # бит без голосов НЕ ставим (иначе короткие тексты
            out |= 1 << i  # слипаются в ложные дубли)
    return out


@dataclass
class Candidate:
    idx: int          # индекс записи в таблице памяти
    text: str         # исходный текст эпизода
    raw_cos: float    # сырое косинусное сходство с запросом
    conf: float       # калиброванная уверенность (sigmoid(a*raw + b))


@dataclass
class RecallResult:
    familiarity: float        # глобальная знакомость темы
    zone: str                 # unknown | unsure | ambiguous | confident
    candidates: list = field(default_factory=list)
    margin: float = 0.0       # запас топ-1 над топ-2

    @property
    def top(self):
        return self.candidates[0] if self.candidates else None


class MemoryTable:
    def __init__(self):
        self.texts: list[str] = []
        self.vecs: list[np.ndarray] = []
        self.active: list[bool] = []
        self.meta: list[dict | None] = []
        self.sims: list[int] = []            # simhash текстов (дедуп)
        self._sim_set: set[int] = set()
        self._mat: np.ndarray | None = None  # кэш матрицы (float32, RAM)

    def __len__(self) -> int:
        return len(self.texts)

    def store(self, text: str, vec: np.ndarray, meta: dict | None = None,
              sim: int | None = None) -> int:
        self.texts.append(text)
        self.vecs.append(np.asarray(vec, dtype=np.float32))
        self.active.append(True)
        self.meta.append(meta)
        s = simhash64(text) if sim is None else sim
        self.sims.append(s)
        self._sim_set.add(s)
        self._mat = None  # инвалидируем кэш
        return len(self.texts) - 1

    def has_text(self, text: str) -> bool:
        """Есть ли точный/почти точный дубликат (по simhash)."""
        return simhash64(text) in self._sim_set

    def set_active(self, idx: int, flag: bool) -> None:
        self.active[idx] = bool(flag)

    def sources(self) -> dict:
        """Источник -> число активных эпизодов."""
        out: dict[str, int] = {}
        for a, m in zip(self.active, self.meta):
            if a and m and m.get("source"):
                out[m["source"]] = out.get(m["source"], 0) + 1
        return out

    def matrix(self) -> np.ndarray:
        """Кэшированная матрица всех векторов (float32, для BLAS-topk)."""
        if self._mat is None or self._mat.shape[0] != len(self.vecs):
            self._mat = (np.stack(self.vecs) if self.vecs
                         else np.zeros((0, 1), dtype=np.float32)).astype(np.float32)
        return self._mat


    def topk(self, vec: np.ndarray, k: int, source: str | None = None):
        """top-k по косинусу; source — фильтр по метаданным (подстрока пути)."""
        if not self.vecs:
            return []
        sims = self.matrix() @ vec
        if source is not None:
            mask = np.array([bool(m and source in str(m.get("source", "")))
                             for m in self.meta], dtype=bool)
            sims = np.where(mask, sims, -np.inf)
        k = min(k, len(self.vecs))
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]
