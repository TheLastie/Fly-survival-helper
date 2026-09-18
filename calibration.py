"""Калибровка уверенности: сырое косинусное сходство -> вероятность правоты.

conf = sigmoid(a * raw_cos + b); параметры a, b подгоняются градиентным
спуском по логистической потере на парах (raw_cos, верно/неверно),
собранных в capacity.py (hold-out эпизоды).
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def apply_calib(raw, a: float, b: float):
    return sigmoid(a * np.asarray(raw, dtype=np.float64) + b)


def fit_logistic(raw, labels, iters: int = 2000, lr: float = 5.0,
                 a0: float = 10.0, b0: float = -7.0):
    """Простой градиентный спуск по log-loss. raw, labels — одномерные массивы."""
    x = np.asarray(raw, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(x) == 0 or np.ptp(x) == 0:
        return a0, b0
    a, b = a0, b0
    for _ in range(iters):
        p = sigmoid(a * x + b)
        ga = float(np.mean((p - y) * x))
        gb = float(np.mean(p - y))
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)
