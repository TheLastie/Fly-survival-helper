"""Генерация ответа поверх извлечённых воспоминаний.

Мозг мухи не генерирует текст — он узнаёт и ранжирует. Этот модуль
превращает RecallResult в готовый ответ пользователю.

Бэкенды (бюджет проекта <= 400 МБ, всё быстро и на CPU):
  TemplateGenerator — по умолчанию: 0 МБ, микросекунды, детерминировано.
                      Ответ собирается из самих эпизодов (они и есть факты);
                      зона уверенности задаёт формулировку.
  OllamaGenerator   — опционально: маленькая локальная LLM через Ollama
                      (например qwen2.5:0.5b). Модель хранится ВНЕ проекта
                      (в ~/.ollama), на вес проекта не влияет. Сервер
                      недоступен — тихий fallback на шаблоны.
"""
from memory import RecallResult


class TemplateGenerator:
    """Собирает ответ из эпизодов памяти, сохраняя честность зон уверенности."""

    def generate(self, query: str, res: RecallResult) -> str:
        if res.zone == "unknown" or res.top is None:
            return ("Похоже, я об этом ещё не знаю. Расскажи — запомню "
                    "с первого раза (скажи «запомни: ...»).")
        top = res.top
        if res.zone == "unsure":
            return (f"Кажется, я это помню: {top.text}. "
                    f"Но уверен всего на {top.conf:.0%} — скажи «верно», "
                    f"и я закреплю, или поправь меня.")
        if res.zone == "ambiguous":
            opts = " ; ".join(f"«{c.text}»" for c in res.candidates[:2])
            return (f"У меня в памяти два похожих варианта: {opts}. "
                    f"Уточни вопрос — и я отвечу точнее.")
        return top.text  # confident: эпизод и есть ответ


class OllamaGenerator:
    """LLM через локальный Ollama (localhost:11434). Модель — вне проекта."""

    def __init__(self, model: str = "qwen2.5:0.5b",
                 host: str = "http://127.0.0.1:11434", timeout: int = 15):
        self.model, self.host, self.timeout = model, host, timeout
        self._template = TemplateGenerator()
        self._ok = self._probe()

    @property
    def available(self) -> bool:
        return self._ok

    def _probe(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def generate(self, query: str, res: RecallResult) -> str:
        if not self._ok:
            return self._template.generate(query, res)
        import json
        import urllib.request
        if res.candidates:
            facts = "\n".join(f"- {c.text}" for c in res.candidates)
        else:
            facts = "(память пуста — фактов по теме нет)"
        prompt = (
            "Ты — ассистент с долговременной памятью. Ниже факты, которые ты помнишь.\n"
            f"{facts}\n\n"
            f"Вопрос: {query}\n"
            "Ответь коротко (1-2 предложения), опираясь только на эти факты. "
            "Если фактов нет — честно скажи, что не знаешь.\nОтвет:"
        )
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps({"model": self.model, "prompt": prompt,
                             "stream": False,
                             "options": {"temperature": 0.2, "num_predict": 120}}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())["response"].strip()
        except Exception:
            self._ok = False
            return self._template.generate(query, res)


SYSTEM_PROMPT = (
    "Ты — ассистент по выживанию в дикой природе. Отвечай ТОЛЬКО на основе "
    "приведённых карточек видов. Правила: (1) не выдумывай факты, которых "
    "нет в карточках; (2) если карточки не содержат ответа — скажи ровно "
    "'В моих карточках нет этой информации — не рискуй' и остановись; "
    "(3) упоминай ядовитость, если она указана; (4) ответ — не более "
    "4 предложений, на языке вопроса.")


class LlamaServerGenerator:
    """Локальная LLM через llama.cpp (llama-server), OpenAI-совместимо.

    Тот же интерфейс, что и у остальных генераторов: recall-контекст
    подаётся в системный промпт с жёстким запретом домыслов.
    Сервер не запущен — тихий fallback на шаблоны."""

    def __init__(self, host: str = "http://127.0.0.1:8081", timeout: int = 60):
        self.host, self.timeout = host, timeout
        self._template = TemplateGenerator()
        self._ok = self._probe()

    @property
    def available(self):
        return self._ok

    def _probe(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(self.host + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def generate(self, query: str, res) -> str:
        if not self._ok:
            return self._template.generate(query, res)
        import json
        import urllib.request
        # ОДНА карточка (топ-1): с многокарточным контекстом микро-LLM
        # путают атрибуцию (токсины «переезжают» между видами) и ложно
        # отказывают; с одной — 4/4 верно (измерено, EVALS).
        if res.candidates:
            facts = res.candidates[0].text
        else:
            facts = "(карточек по теме нет)"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nКарточки:\n" + facts},
            {"role": "user", "content": query},
        ]
        req = urllib.request.Request(
            self.host + "/v1/chat/completions",
            data=json.dumps({"messages": msgs, "temperature": 0.1,
                             "max_tokens": 220}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"]
            return out.strip()
        except Exception:
            self._ok = False
            return self._template.generate(query, res)


def make_generator(backend: str = "template", ollama_model: str = "qwen2.5:0.5b",
                   llama_host: str = "http://127.0.0.1:8081"):
    """Фабрика с тихим fallback на шаблоны, если Ollama недоступна."""
    if backend == "ollama":
        gen = OllamaGenerator(model=ollama_model)
        if gen.available:
            return gen
        print("(ollama недоступна — генератор: шаблоны)")
    if backend == "llama":
        gen = LlamaServerGenerator(host=llama_host)
        if gen.available:
            print(f"(llama-server: {llama_host})")
            return gen
        print("(llama-server не отвечает — генератор: шаблоны)")
    return TemplateGenerator()
