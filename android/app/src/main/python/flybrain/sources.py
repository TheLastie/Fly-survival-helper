"""Универсальные адаптеры источников: файл/папка -> список документов {text, meta}.

Покрывает: тексты и разметку (txt, md, rst, html...), исходный код,
таблицы (csv — каждая строка документ), структурированные данные
(json/jsonl — каждая запись документом). PDF — опционально через pypdf
(если не установлен, файл пропускается с сообщением).
"""
import csv
import json
import os
import re

TEXT_EXTS = {".txt", ".md", ".rst", ".py", ".rs", ".go", ".js", ".ts", ".jsx",
             ".tsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".html", ".htm",
             ".css", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh",
             ".sql", ".log", ".env"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             "dist", "build", ".idea", ".vscode"}
MAX_FILE_MB = 20

# Порядок важен: строгие кодировки первыми, latin-1 — аварийный (не падает никогда).
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "koi8-r", "mac-cyrillic", "latin-1")


def read_text(path: str) -> str:
    """Прочитать текст с автоопределением кодировки (royallib/cp1251 и т.п.)."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def read_file(path: str) -> list:
    ext = os.path.splitext(path)[1].lower()
    meta = {"source": os.path.abspath(path), "ext": ext}
    try:
        if ext in TEXT_EXTS:
            return [{"text": read_text(path), "meta": meta}]
        if ext == ".csv":
            rows = list(csv.reader(read_text(path).splitlines()))
            if not rows:
                return []
            header, docs = rows[0], []
            for i, row in enumerate(rows[1:], 1):
                text = _clean(" | ".join(f"{h}: {c}" for h, c in zip(header, row) if c))
                if text:
                    docs.append({"text": text, "meta": {**meta, "row": i}})
            return docs
        if ext in (".jsonl", ".ndjson"):
            docs = []
            for i, line in enumerate(read_text(path).splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docs.append({"text": _clean(json.dumps(obj, ensure_ascii=False)),
                             "meta": {**meta, "line": i}})
            return docs
        if ext == ".json":
            obj = json.loads(read_text(path))

            def flatten(o, p=""):
                if isinstance(o, dict):
                    for k, v in o.items():
                        yield from flatten(v, f"{p}.{k}" if p else str(k))
                elif isinstance(o, list):
                    for j, v in enumerate(o):
                        yield from flatten(v, f"{p}[{j}]")
                else:
                    yield p, str(o)

            return [{"text": _clean(f"{k}: {v}"), "meta": meta} for k, v in flatten(obj)]
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                print(f"(pdf пропущен — установите pypdf: {path})")
                return []
            reader = PdfReader(path)
            return [{"text": p.extract_text() or "", "meta": {**meta, "page": n}}
                    for n, p in enumerate(reader.pages, 1)]
    except Exception as e:  # поврежденный файл — не роняем индексацию
        print(f"(не прочитан {path}: {e})")
    return []


def iter_files(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in sorted(fns):
            yield os.path.join(dp, fn)


def load_path(root: str, max_file_mb: int = MAX_FILE_MB) -> list:
    """Все читаемые документы из файла или дерева папок."""
    docs = []
    for path in iter_files(root):
        try:
            if os.path.getsize(path) > max_file_mb * 1024 * 1024:
                continue
        except OSError:
            continue
        docs.extend(read_file(path))
    return docs
