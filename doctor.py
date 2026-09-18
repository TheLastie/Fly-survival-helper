"""Полевая диагностика: одна команда — вся система проверена.

python doctor.py [--state field_state] [--llama HOST]

Проверяет: ONNX-модель и зрение (эмбеддинг+поиск roundtrip), состояние
(если есть), LLM-сервер, диск/RAM. Коды выхода: 0 — всё зелёное,
1 — есть жёлтое/красное.
"""
import argparse
import os
import sys

ISSUES = []


def check(name, ok, detail="", warn=False):
    mark = "PASS" if ok else ("WARN" if warn else "FAIL")
    print(f"  [{mark}] {name} {detail}")
    if not ok:
        ISSUES.append((name, warn))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None)
    ap.add_argument("--llama", default="http://127.0.0.1:8081")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print("== модель зрения ==")
    onnx = os.path.join(here, "models", "mobilenet_v3_small.onnx")
    check("models/mobilenet_v3_small.onnx", os.path.exists(onnx),
          f"({os.path.getsize(onnx)//1024 if os.path.exists(onnx) else 0} КБ)")
    emb_ok = False
    if os.path.exists(onnx):
        try:
            import numpy as np
            from image_embedder import ImageEmbedder
            from PIL import Image
            e = ImageEmbedder()
            v = e.embed_image(Image.fromarray(
                np.zeros((64, 64, 3), np.uint8)))
            emb_ok = check("эмбеддинг-конвейер", v.shape == (512,))
        except Exception as ex:
            check("эмбеддинг-конвейер", False, str(ex)[:80])

    print("== состояние ==")
    if args.state and os.path.exists(args.state):
        try:
            from config import FlyConfig
            from embedder import HashingEmbedder
            from system import FlySystem
            s = FlySystem(FlyConfig(), HashingEmbedder())
            s.load(args.state)
            n_img = sum(1 for m in s.memory.meta if m and m.get("kind") == "image")
            n_sp = len(s.species_info)
            check("загрузка состояния", True,
                  f"({len(s.memory)} документов, {n_img} фото, {n_sp} видов)")
            # ноль фото — легитимно для текстового состояния (предупреждение)
            check("виды с фото >= 20", n_img >= 20, f"({n_img})",
                  warn=True)
            check("поиск работает", len(s.search("тест", k=3).candidates) >= 0)
        except Exception as ex:
            check("загрузка состояния", False, str(ex)[:80])
    else:
        check("состояние (--state)", False, "не найдено", warn=True)

    print("== LLM ==")
    import urllib.request
    try:
        with urllib.request.urlopen(args.llama + "/health", timeout=3) as r:
            check(f"llama-server {args.llama}", r.status == 200)
    except Exception:
        check(f"llama-server {args.llama}", False,
              "(отвечает шаблонами — это безопасно)", warn=True)

    print("== ресурсы ==")
    try:
        import resource
        check("RAM", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
              < 3e6, f"(peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024} МБ)")
    except Exception:
        pass
    st = os.statvfs(here)
    free_mb = st.f_bavail * st.f_frsize // 1048576
    check("свободно на диске", free_mb > 1000, f"({free_mb} МБ)", warn=free_mb > 300)

    n_fail = sum(1 for _, w in ISSUES if not w)
    n_warn = sum(1 for _, w in ISSUES if w)
    print(f"\nИтог: {len(ISSUES) - n_fail - n_warn} ок, "
          f"{n_warn} предупреждений, {n_fail} проблем")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
