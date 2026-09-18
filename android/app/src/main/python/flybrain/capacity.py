"""Эксперимент: ёмкость памяти + калибровка уверенности + графики.

Для каждого объема M one-shot записывается M эпизодов, затем три типа проб:
  known    — перефраз записанного факта              -> accuracy@top-1
  hard_neg — перефраз НЕзаписанного факта того же
             шаблона (та же тема, другая сущность)   -> доля ложно-уверенных
  geo      — вопрос из другой предметной области     -> знакомость новизны

Пары (raw_cos top-1, верно/неверно) с known и hard_neg используются
для подгонки калибровки conf = sigmoid(a*raw + b) и подбора порогов.

Запуск: python capacity.py                    (полный прогон)
        python capacity.py --ms 500 2000 5000 --probes 150   (быстрый)
"""
import argparse
import json
import os

import numpy as np

from calibration import fit_logistic
from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem

NAMES = ["Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Попов", "Волков",
         "Соколов", "Лебедев", "Козлов", "Новиков", "Морозов", "Петрова", "Васильева",
         "Захаров", "Павлов", "Семенов", "Голубев", "Виноградов", "Богданов",
         "Тарасов", "Орлов", "Белов", "Комаров", "Киселев", "Макаров", "Андреев",
         "Ковалев", "Ильин", "Гусев"]
ROLES = ["инженер", "биолог", "стажер", "математик", "техник", "программист",
         "аспирант", "аналитик"]
LABS = ["Альфа", "Бета", "Гамма", "Дельта", "Омега", "Вега", "Сириус", "Орион",
        "Лира", "Цефей", "Персей", "Тукан"]
PROJS = ["радар", "штатив", "калибр", "меридиан", "полюс", "кварц", "град",
         "вихрь", "струна", "флюс", "корунд", "азимут"]

T_FACT = "{name} — {role} лаборатории {lab}, проект {proj}"
T_PARA = "{role} лаборатории {lab} в проекте {proj} — это {name}"
T_PARA2 = "кто {role} из лаборатории {lab} по проекту {proj}? это {name}"

COUNTRIES = ["Франция", "Германия", "Италия", "Испания", "Польша", "Швеция",
             "Норвегия", "Финляндия", "Япония", "Китай", "Индия", "Бразилия",
             "Канада", "Египет", "Турция"]
CITIES = ["Париж", "Берлин", "Рим", "Мадрид", "Варшава", "Стокгольм", "Осло",
          "Хельсинки", "Токио", "Пекин", "Дели", "Бразилиа", "Оттава", "Каир",
          "Анкара"]
T_GEO = "столица {country} — {city}"
T_GEO_PARA = "какая столица у страны {country}? это {city}"


def make_facts(n: int, seed: int = 1):
    """Уникальные факты шаблона лабораторий (пространство ~35k комбинаций)."""
    rng = np.random.default_rng(seed)
    combos = [(a, b, c, d) for a in NAMES for b in ROLES for c in LABS for d in PROJS]
    rng.shuffle(combos)
    facts, paras = [], []
    for name, role, lab, proj in combos[:n]:
        facts.append(T_FACT.format(name=name, role=role, lab=lab, proj=proj))
        tmpl = T_PARA2 if rng.random() < 0.5 else T_PARA
        paras.append(tmpl.format(name=name, role=role, lab=lab, proj=proj))
    return facts, paras


def make_geo(n: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    pairs = [(c, s) for c in COUNTRIES for s in CITIES]
    rng.shuffle(pairs)
    facts, paras = [], []
    for country, city in pairs[:n]:
        facts.append(T_GEO.format(country=country, city=city))
        paras.append(T_GEO_PARA.format(country=country, city=city))
    return facts, paras


def run(ms, probes, n_kc, sparsity, out_png, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = max(ms) + probes + 10
    all_facts, all_paras = make_facts(total, seed)
    geo_facts, geo_paras = make_geo(probes + 10, seed + 1)

    accs, hard_wrong, fam_known, fam_hard, fam_geo = [], [], [], [], []
    calib_raw, calib_lab = [], []

    for M in ms:
        cfg = FlyConfig(n_kc=n_kc, sparsity=sparsity)
        sys_ = FlySystem(cfg, HashingEmbedder())
        for i in range(M):
            sys_.memorize(all_facts[i])

        rng = np.random.default_rng(M)
        known_idx = rng.choice(M, size=min(probes, M), replace=False)

        correct = wrong_conf = 0
        f_known = []
        for i in known_idx:
            res = sys_.ask(all_paras[i], k=1)
            ok = res.top is not None and res.top.text == all_facts[i]
            correct += int(ok)
            f_known.append(res.familiarity)
            calib_raw.append(res.top.raw_cos if res.top else 0.0)
            calib_lab.append(int(ok))
        f_hard = []
        for j in range(M, M + probes):
            res = sys_.ask(all_paras[j], k=1)
            wrong = res.top is not None and res.top.text != all_facts[j]
            wrong_conf += int(wrong and res.zone == "confident")
            f_hard.append(res.familiarity)
            calib_raw.append(res.top.raw_cos if res.top else 0.0)
            calib_lab.append(0)
        f_geo = [sys_.ask(geo_paras[j]).familiarity for j in range(probes)]

        accs.append(correct / len(known_idx))
        hard_wrong.append(wrong_conf / probes)
        fam_known.append(float(np.mean(f_known)))
        fam_hard.append(float(np.mean(f_hard)))
        fam_geo.append(float(np.mean(f_geo)))
        print(f"M={M:6d}  acc@1={accs[-1]:.3f}  false_conf={hard_wrong[-1]:.3f}  "
              f"fam_known={fam_known[-1]:.3f}  fam_hard={fam_hard[-1]:.3f}  fam_geo={fam_geo[-1]:.3f}")

    a, b = fit_logistic(calib_raw, calib_lab)
    theta_low = float((np.mean(fam_geo) + np.mean(fam_known)) / 2)
    print(f"\nкалибровка: a={a:.2f}, b={b:.2f}  (conf = sigmoid(a*raw + b))")
    print(f"рекомендуемый theta_low ≈ {theta_low:.2f}")

    with open(os.path.splitext(out_png)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"calib_a": a, "calib_b": b, "theta_low": theta_low,
                   "ms": ms, "acc": accs, "false_conf": hard_wrong,
                   "fam_known": fam_known, "fam_hard": fam_hard, "fam_geo": fam_geo},
                  f, ensure_ascii=False, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ms, accs, "o-", color="tab:blue", label="known recall @top-1")
    ax1.plot(ms, hard_wrong, "s--", color="tab:red", label="hard neg: false confident")
    ax1.set_xlabel("episodes stored (M)")
    ax1.set_ylabel("rate")
    ax1.set_title("Memory capacity (one-shot learning)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ms, fam_known, "o-", color="tab:green", label="known")
    ax2.plot(ms, fam_hard, "s-", color="tab:orange", label="hard negative")
    ax2.plot(ms, fam_geo, "^--", color="tab:red", label="novel domain")
    ax2.axhline(theta_low, color="gray", ls=":", lw=1)
    ax2.set_xlabel("episodes stored (M)")
    ax2.set_ylabel("familiarity")
    ax2.set_title("Familiarity by probe type")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"график сохранен: {out_png}")
    return accs, fam_known, fam_hard, fam_geo, (a, b, theta_low)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+",
                    default=[500, 1000, 2500, 5000, 10000, 20000])
    ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--n-kc", type=int, default=2000)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "capacity_curve.png"))
    args = ap.parse_args()
    run(args.ms, args.probes, args.n_kc, args.sparsity, args.out)
