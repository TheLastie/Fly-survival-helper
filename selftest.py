"""Самопроверка ядра + сценарное демо «запомни -> спроси перефразом -> оцени -> проверь».

Запуск: python selftest.py
"""
import tempfile

import numpy as np

from calibration import apply_calib, fit_logistic
from config import FlyConfig
from embedder import HashingEmbedder
from system import FlySystem

F1 = "Морган открыл наследование белоглазки у дрозофилы"
F2 = "Павлов описал условный рефлекс у собаки"
F3 = "Левенгук увидел микроорганизмы в капле воды"
Q1 = "кто открыл наследование белоглазки?"
Q2 = "кто описал условный рефлекс у собаки?"
Q3 = "кто увидел микроорганизмы в капле воды?"
NOVEL = "какая столица Франции?"

G1 = "Иванов работает инженером в лаборатории Каспий"
G2 = "Петров работает инженером в лаборатории Каспий"
GQ = "кто работает инженером в лаборатории Каспий"

results = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def demo():
    print("=== Демо: диалог с мозгом ===")
    sys_ = FlySystem(FlyConfig(), HashingEmbedder())
    from cli import format_answer
    for fact in (F1, F2, F3):
        st = sys_.memorize(fact)
        print(f">  запомни: {fact}")
        print(f"<  запомнил (знакомость {st['familiarity_after']:.2f})")
    for q in (Q1, Q2, Q3, NOVEL):
        res = sys_.ask(q)
        print(f">  {q}")
        print("<  " + format_answer(res, sys_.cfg).replace("\n", "\n<  "))
    # наказание
    res = sys_.ask(Q1)
    print(f">  {Q1}")
    print(f"<  {res.top.text} (уверенность {res.top.conf:.0%})")
    st = sys_.feedback(-1)
    print(f">  забудь")
    print(f"<  DAN -1: знакомость упала до {st['familiarity_after']:.2f}")
    res = sys_.ask(Q1)
    print(f">  {Q1}")
    print(f"<  " + format_answer(res, sys_.cfg).replace("\n", "\n<  "))
    # соседняя память не пострадала
    res2 = sys_.ask(Q2)
    print(f">  {Q2}")
    print(f"<  " + format_answer(res2, sys_.cfg).replace("\n", "\n<  "))
    return sys_


def tests():
    print("=== Тесты ядра ===")
    cfg = FlyConfig()
    emb = HashingEmbedder()
    sys_ = FlySystem(cfg, emb)

    # 1. ортогональность KC-кодов: среднее пересечение случайных пар
    rng = np.random.default_rng(0)
    codes = [sys_.brain.sparse_code(rng.normal(size=cfg.d_model)) for _ in range(50)]
    codes = [c for c in codes if c.sum() > 0]
    overlaps = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            overlaps.append(float((codes[i] * codes[j]).sum()))
    mean_ov = float(np.mean(overlaps))
    k_eff = float(np.mean([c.sum() for c in codes]))
    check("ортогональность KC-кодов", mean_ov < 2.0 * (k_eff ** 2) / cfg.n_kc,
          f"mean overlap={mean_ov:.1f}, k_eff={k_eff:.0f}, ожидание ~{(k_eff**2)/cfg.n_kc:.1f}")

    # 2. one-shot recall перефразом
    for f in (F1, F2, F3):
        sys_.memorize(f)
    r1 = sys_.ask(Q1)
    check("one-shot recall (перефраз)", r1.top is not None and r1.top.text == F1 and r1.zone != "unknown",
          f"zone={r1.zone}, fam={r1.familiarity:.2f}, conf={r1.top.conf:.0%}")
    check("знакомость выше порога", r1.familiarity >= cfg.theta_low, f"fam={r1.familiarity:.2f}")

    # 3. новизна: чужой вопрос -> unknown
    rn = sys_.ask(NOVEL)
    check("новизна (unknown)", rn.zone == "unknown", f"zone={rn.zone}, fam={rn.familiarity:.2f}")

    # 4. наказание стирает, соседи целы
    fam_before = {F2: sys_.brain.familiarity(emb.embed(Q2)),
                  F3: sys_.brain.familiarity(emb.embed(Q3))}
    sys_.ask(Q1)
    st = sys_.feedback(-1)
    fam_a = sys_.brain.familiarity(emb.embed(Q1))
    check("наказание снизило знакомость", fam_a < st["familiarity_before"],
          f"{st['familiarity_before']:.2f} -> {fam_a:.2f}")
    r_after = sys_.ask(Q1)
    check("наказанный эпизод не возвращается", r_after.top is None or r_after.top.text != F1,
          f"zone={r_after.zone}")
    ok_neighbors = all(sys_.brain.familiarity(emb.embed(q)) >= f - 0.05
                       for q, f in ((Q2, fam_before[F2]), (Q3, fam_before[F3])))
    check("соседние памяти не пострадали", ok_neighbors)

    # 5. затухание: память слабеет от чужих записей (decay)
    cfg2 = FlyConfig(decay=0.01, seed=7)
    sys2 = FlySystem(cfg2, HashingEmbedder())
    sys2.memorize(F1)
    fam_right = sys2.brain.familiarity(sys2.embedder.embed(F1))
    for i in range(300):
        sys2.memorize(f"служебная запись номер {i} про разное содержимое")
    fam_later = sys2.brain.familiarity(sys2.embedder.embed(F1))
    check("гомеостатическое затухание", fam_later < fam_right,
          f"{fam_right:.2f} -> {fam_later:.2f} после 300 чужих записей")

    # 6. калибровка: fit на синтетике дает монотонную раскладку в (0,1)
    raw = np.concatenate([np.random.uniform(0.75, 0.95, 200), np.random.uniform(0.5, 0.72, 200)])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    a, b = fit_logistic(raw, labels)
    c_hi = float(apply_calib(0.9, a, b)); c_lo = float(apply_calib(0.55, a, b))
    check("калибровка монотонна и в (0,1)", 0.9 < c_hi <= 1.0 and 0.0 < c_lo < 0.3,
          f"a={a:.1f}, b={b:.1f}, conf(0.9)={c_hi:.2f}, conf(0.55)={c_lo:.2f}")

    # 7. неоднозначность: два почти одинаковых факта
    cfg3 = FlyConfig(seed=3)
    sys3 = FlySystem(cfg3, HashingEmbedder())
    sys3.memorize(G1)
    sys3.memorize(G2)
    rg = sys3.ask(GQ)
    check("неоднозначность (ambiguous)", rg.zone in ("ambiguous", "confident"),
          f"zone={rg.zone}, margin={rg.margin:.3f}, fam={rg.familiarity:.2f}")
    check("запас мал", rg.margin < 0.06, f"margin={rg.margin:.3f}")

    # 8. сохранение/загрузка
    with tempfile.TemporaryDirectory() as td:
        sys_.save(td)
        cfg4 = FlyConfig()
        sys4 = FlySystem(cfg4, HashingEmbedder())
        sys4.load(td)
        f_q = sys_.brain.familiarity(emb.embed(Q2))
        f_l = sys4.brain.familiarity(sys4.embedder.embed(Q2))
        check("save/load roundtrip", abs(f_q - f_l) < 1e-5, f"fam {f_q:.4f} vs {f_l:.4f}")

    # 9. генератор: template честно отражает зону; ollama без сервера -> fallback
    from generator import OllamaGenerator, TemplateGenerator
    gen = TemplateGenerator()
    r_ok = sys_.ask(Q2)
    txt = gen.generate(Q2, r_ok)
    check("генератор (template)", "Павлов" in txt, f"zone={r_ok.zone}, fam={r_ok.familiarity:.2f}")
    g_oll = OllamaGenerator(timeout=3)
    txt_fb = g_oll.generate(Q2, r_ok)
    check("ollama fallback на шаблоны", txt_fb == txt and not g_oll.available)
    res_ans, text_ans = sys_.answer(Q3)
    check("system.answer", "Левенгук" in text_ans, f"zone={res_ans.zone}")

    # 10. длинный текст: нарезка на эпизоды + детальные вопросы
    from loader import load_text
    LONG = (
        "Дрозофила меланогастер стала модельным организмом в начале двадцатого века. "
        "Томас Морган начал работать с ней в 1907 году в лаборатории Колумбийского университета. "
        "Муху выбрали за короткий цикл развития: всего десять дней от яйца до взрослой особи. "
        "За год дрозофила даёт до двадцати пяти поколений. "
        "В 1910 году Морган обнаружил мутанта с белыми глазами вместо красных. "
        "Эксперименты с белоглазкой привели к открытию сцепления генов с полом. "
        "Морган сформулировал хромосомную теорию наследственности в 1915 году. "
        "Вместе с ним работали студенты: Стёртевант, Мёллер и Бриджес. "
        "Стёртевант построил первую генетическую карту в 1913 году. "
        "Мёллер показал, что рентгеновские лучи вызывают мутации, в 1927 году. "
        "В 1933 году Морган получил Нобелевскую премию по физиологии и медицине. "
        "Дрозофила имеет четыре пары хромосом. "
        "Геном дрозофилы секвенирован в 2000 году. "
        "В мозге мухи около ста тысяч нейронов. "
        "Грибовидное тело занимает значительную часть мозга насекомого и отвечает за обучение и память."
    )
    cfg5 = FlyConfig(seed=5)
    sys5 = FlySystem(cfg5, HashingEmbedder())
    n_ep = len(load_text(sys5, LONG, source="история дрозофилы"))
    check("loader: текст нарезан на эпизоды", n_ep >= 10, f"эпизодов: {n_ep}")
    qa = [("в каком году Морган обнаружил белоглазку?", "1910"),
          ("кто построил первую генетическую карту?", "Стёртевант"),
          ("сколько пар хромосом у дрозофилы?", "четыре")]
    ok_all, demo_lines = True, []
    for q, key in qa:
        res = sys5.ask(q, k=3)
        texts = [c.text for c in res.candidates]
        hit = any(key in t for t in texts)
        ok_all &= hit
        demo_lines.append(f"     > {q}\n"
                          f"     < fam={res.familiarity:.2f} зона={res.zone} | "
                          f"top: {texts[0][:70] if texts else '—'}")
    print("\n".join(demo_lines))
    check("длинный текст: детали находятся в top-3", ok_all)


if __name__ == "__main__":
    demo()
    print()
    tests()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\nИтог: {len(results) - n_fail}/{len(results)} тестов пройдено.")
    raise SystemExit(1 if n_fail else 0)
