"""Точка входа Python в APK (Chaquopy) — самодостаточный запуск.

Первый запуск: распаковка assets/state (готовый field_state: 4027
документов, 117 видов) в filesDir → load → Web-UI на 127.0.0.1:8321.
Никаких индексаций и обязательных загрузок: всё внутри одного файла APK.

LLM (опционально): если рядом с файлами приложения есть libllama.so
(assets при сборке, см. README_ANDROID.md) и qwen15b-q4.gguf — включаем
локальную LLM через ctypes (llama_local.py). Иначе — безопасные шаблоны.
"""
import os
import shutil
import sys
import threading

_server = None
_sys = None


def _copy_tree_assets(mgr, src, dst):
    """Рекурсивная копия папки из assets (файлы + подпапки)."""
    os.makedirs(dst, exist_ok=True)
    for name in mgr.list(src):
        s = src + "/" + name
        d = os.path.join(dst, name)
        is_dir = False
        try:
            is_dir = len(mgr.list(s)) > 0
        except Exception:
            is_dir = False
        if is_dir:
            _copy_tree_assets(mgr, s, d)
        else:
            with open(d, "wb") as f:
                f.write(mgr.open(s).read())


def _setup_state(files_dir):
    """Состояние уже скопировано Kotlin-ом в filesDir/state."""
    dst = os.path.join(files_dir, "state")
    return dst if os.path.exists(os.path.join(dst, "manifest.json")) else None


def _setup_llm(files_dir):
    """libllama.so + модель из assets/filesDir -> LlamaLocal или None."""
    try:
        import llama_local
        here = os.path.dirname(os.path.abspath(__file__))
        lib = os.path.join(here, "libllama.so")
        model = os.path.join(files_dir, "qwen15b-q4.gguf")
        if os.path.exists(lib) and os.path.exists(model):
            return llama_local.LlamaLocal(lib, model, n_ctx=2048, n_threads=6)
    except Exception:
        pass
    return None


def _crash_server(files_dir, tb):
    """Аварийный сервер: показывает текст ошибки в WebView."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    page = ("<meta charset=utf-8><body style='font-family:monospace;"
            "padding:16px;white-space:pre-wrap'>"
            "<h2>FlyBrain: ошибка запуска</h2>Сообщите этот текст разработчику:"
            "\n\n" + tb.replace("&", "&amp;").replace("<", "&lt;")
            + "</body>").encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 8321), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start(files_dir):
    global _server, _sys
    import sys
    import traceback
    try:
        _start_inner(files_dir)
    except Exception:
        tb = traceback.format_exc()
        with open(os.path.join(files_dir, "crash.log"), "w", encoding="utf-8") as f:
            f.write(tb)
        _server = _crash_server(files_dir, tb)


def _start_inner(files_dir):
    os.environ.setdefault("FLYBRAIN_MODELS_DIR",
                          os.path.join(files_dir, "models"))
    proj = os.path.dirname(os.path.abspath(__file__))
    for p in (proj, os.path.join(proj, "flybrain")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from config import FlyConfig
    from embedder import HashingEmbedder
    from system import FlySystem

    _sys = FlySystem(FlyConfig(), HashingEmbedder(), gen_backend="template")

    state = _setup_state(files_dir)
    if state:
        try:
            _sys.load(state)
        except Exception:
            state = None
    if not state:
        # запасной путь: индексация assets/dbs (если state не вшит)
        dbs = os.path.join(files_dir, "dbs")
        if os.path.isdir(dbs):
            for name in sorted(os.listdir(dbs)):
                path = os.path.join(dbs, name)
                if os.path.isdir(path):
                    try:
                        _sys.index_species(path)
                    except Exception:
                        pass
            try:
                _sys.save(os.path.join(files_dir, "state"))
            except Exception:
                pass

    # LLM: локальная (ctypes libllama) или шаблоны
    llm = _setup_llm(files_dir)
    if llm is not None:
        from generator import LlamaLocalGenerator
        _sys.generator = LlamaLocalGenerator(llm)

    import webui
    webui.Handler.system = _sys
    from http.server import ThreadingHTTPServer
    _server = ThreadingHTTPServer(("127.0.0.1", 8321), webui.Handler)
    threading.Thread(target=_server.serve_forever, daemon=True).start()


def stop():
    global _server
    if _server:
        _server.shutdown()
        _server = None
