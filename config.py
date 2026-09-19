"""Параметры системы «мозг мухи».

Все пороги (theta_low, theta_mid, calib_*) зависят от эмбеддера.
Скрипт capacity.py подбирает их эмпирически и сохраняет в calib.json.
"""
from dataclasses import dataclass


@dataclass
class FlyConfig:
    # --- анатомия ---
    n_kc: int = 2000          # число клеток Кеньона
    sparsity: float = 0.05    # базовая доля активных KC (top-k)
    d_model: int = 512        # размерность эмбеддинга (переопределяется эмбеддером)

    # --- пластичность ---
    lr_pos: float = 0.3       # скорость обучения при награде
    lr_neg_scale: float = 1.0  # lr_neg = lr_pos * scale; чистый хеббовский след:
                               # наказание вычитает ровно тот же вектор — соседи не страдают
    decay: float = 0.0005     # гомеостатическое затухание связей KC->MBON
    soft_coef: float = 0.9    # мягкий порог в sparse_code: добавляем KC с pre >= soft_coef * k-го значения

    # --- память / ответ ---
    topk_mem: int = 3         # сколько кандидатов показывать из таблицы памяти
    rerank_pool: int = 30     # пул RRF-кандидатов для LTR-переранжирования
    rm3_enabled: bool = True  # RM3: экспансия запроса терминами топ-5
    rm3_n: int = 8            # число экспансионных терминов
    rm3_w: float = 0.4        # их вес в повторном проходе
    session_n: int = 8        # глубина сессионного контекста
    session_w: float = 0.3    # вес контекста в векторе запроса

    # --- зоны уверенности ---
    theta_low: float = 0.40   # familiarity (статистика; гейт смотрит theta_conf)
    theta_conf: float = 0.35  # conf топ-1 ниже — «не помню» (калиброванная P(верно))
    recognize_threshold: float = 0.75  # калибровано на 46 видов + OOD-пробах (см. EVALS)
    recognize_margin: float = 0.01  # мин. отрыв от другого вида для вердикта
    theta_mid: float = 0.75   # conf ниже — «кажется, но не уверен»
    delta_margin: float = 0.02  # запас топ-1 над топ-2 меньше — «неоднозначно»

    # --- калибровка уверенности: conf = sigmoid(a * raw_cos + b) ---
    calib_a: float = 38.7    # подобрано capacity.py для HashingEmbedder
    calib_b: float = -28.0
    # калибровка для BM25-only кандидатов: conf = sigmoid(a2 * bm25_norm + b2)
    calib_a2: float = 4.0
    calib_b2: float = -2.0

    seed: int = 42

    @property
    def k_active(self) -> int:
        return max(1, int(self.n_kc * self.sparsity))

    @property
    def lr_neg(self) -> float:
        return self.lr_pos * self.lr_neg_scale
