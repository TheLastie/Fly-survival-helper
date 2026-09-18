"""Ядро: грибовидное тело дрозофилы.

PN -> KC  : фиксированная случайная проекция + разреженная бинарная активация.
KC -> MBON: единственный обучаемый слой, трехфакторная пластичность:
            dW = lr * reward * outer(kc, x - out),  плюс гомеостатическое затухание.

Оптимизация: decay применяется «лениво» через глобальный множитель _scale,
обновляются только строки W_kc_mbon, соответствующие активным KC
(вместо полной матрицы n_kc x d_model на каждом шаге).
"""
import numpy as np

from config import FlyConfig


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


class FlyBrain:
    def __init__(self, cfg: FlyConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        # PN -> KC: фиксированные случайные веса
        self.W_pn_kc = rng.normal(
            0.0, 1.0 / np.sqrt(cfg.d_model), size=(cfg.d_model, cfg.n_kc)
        ).astype(np.float32)
        # KC -> MBON: обучаемые, инициализация нулями
        self.W_kc_mbon = np.zeros((cfg.n_kc, cfg.d_model), dtype=np.float32)
        self._scale = 1.0  # ленивое затухание: эффективные веса = W_kc_mbon * _scale

    # --- разреженное кодирование ---
    def sparse_code(self, x: np.ndarray) -> np.ndarray:
        pre = x @ self.W_pn_kc
        k = self.cfg.k_active
        top = np.argpartition(-pre, k - 1)[:k]
        code = np.zeros(self.cfg.n_kc, dtype=np.float32)
        thr = float(pre[top[-1]])
        if self.cfg.soft_coef is not None:
            extra = np.where(pre >= self.cfg.soft_coef * thr)[0]
            top = np.union1d(top, extra)
        code[top] = 1.0
        return code

    def active_rows(self, x: np.ndarray) -> np.ndarray:
        return np.flatnonzero(self.sparse_code(x))

    # --- отклик памяти ---
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Восстановленный вектор out = kc @ W (эффективный)."""
        rows = self.active_rows(x)
        if rows.size == 0:
            return np.zeros(self.cfg.d_model, dtype=np.float32)
        return self._scale * self.W_kc_mbon[rows].sum(axis=0)

    def familiarity(self, x: np.ndarray) -> float:
        """Знакомость = cos(out, x) * сила следа.

        Сила следа = min(1, ||out|| / (lr_pos * k_active)): у свежей записи ~1.
        Гомеостатическое затухание и наказание уменьшают ||out||, и потому
        реально стирают память — сам cos инвариантен к общему масштабу весов,
        без этого множителя затухание было бы невидимым.
        """
        out = self.forward(x)
        c = cos(out, x)
        denom = self.cfg.lr_pos * self.cfg.k_active
        strength = min(1.0, float(np.linalg.norm(out)) / denom) if denom > 0 else 0.0
        return c * strength

    # --- трехфакторное обучение ---
    def learn(self, x: np.ndarray, reward: float = +1.0, weight: float = 1.0) -> dict:
        """reward=+1 — награда (запомнить/усилить), -1 — наказание (стереть).

        Чистый хеббовский след (классическая модель KC->MBON с дофаминовым
        гейтом): награда добавляет x в активные строки, наказание вычитает.
        Ошибка (x - out) намеренно НЕ используется: суммарная реконструкция
        из ~k строк имеет масштаб >> 1 и заражала бы новые записи чужими
        паттернами через разделяемые клетки Кеньона.
        """
        rows = self.active_rows(x)
        fam_before = self.familiarity(x)
        if self.cfg.decay:
            self._scale *= (1.0 - self.cfg.decay)
            if self._scale < 0.5:  # материализуем затухание, чтобы не раздувать W
                self.W_kc_mbon *= self._scale
                self._scale = 1.0
        if rows.size:
            lr = self.cfg.lr_pos if reward >= 0 else self.cfg.lr_neg
            self.W_kc_mbon[rows] += (lr * reward * weight) * x[np.newaxis, :]
        return {"familiarity_before": fam_before, "familiarity_after": self.familiarity(x)}

    # --- сохранение ---
    def state_dict(self) -> dict:
        return {"W_pn_kc": self.W_pn_kc, "W_kc_mbon": self.W_kc_mbon, "scale": self._scale}

    def load_state_dict(self, state: dict) -> None:
        self.W_pn_kc = state["W_pn_kc"]
        self.W_kc_mbon = state["W_kc_mbon"]
        self._scale = float(state["scale"])
