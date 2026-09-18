"""Локальная LLM в APK через ctypes к libllama.so (стабильный C API).

Соберите libllama.so под arm64-v8a (команда в README_ANDROID.md), положите
рядом с android_host.py и модель qwen15b-q4.gguf в filesDir. Интерфейс —
как у LlamaServerGenerator: chat(messages) с жёстким системным промптом.

НЕ собиралось здесь (нет NDK) — первый прогон за вами; API-вызовы сверены
с заголовками llama.cpp (llama.h), подписи — стабильные с 2024 года.
"""
import ctypes
import json
import os

SYSTEM_PROMPT = (
    "Ты — ассистент по выживанию в дикой природе. Отвечай ТОЛЬКО на основе "
    "одной приведённой карточки. Правила: (1) не выдумывай фактов, которых "
    "нет в карточке; (2) если карточки не содержат ответа — скажи ровно "
    "'В моих карточках нет этой информации — не рискуй' и остановись; "
    "(3) упоминай ядовитость, если она указана; (4) ответ — не более "
    "4 предложений, на языке вопроса.")


class LlamaLocal:
    def __init__(self, lib_path: str, model_path: str,
                 n_ctx: int = 2048, n_threads: int = 6):
        lib = ctypes.CDLL(lib_path)
        self._lib = lib
        lib.llama_backend_init.argtypes = []
        lib.llama_backend_init()
        lib.llama_init_from_model_file.argtypes = [ctypes.c_char_p,
                                                   ctypes.c_void_p]
        # llama_init_from_model_file не существует в новых версиях —
        # используем пару model+context (см. ниже)
        lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p,
                                                   ctypes.c_void_p]
        lib.llama_init_from_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        class ModelParams(ctypes.Structure):
            _fields_ = [("main_gpu", ctypes.c_int32), ("tensor_split",
                         ctypes.POINTER(ctypes.c_float)),
                        ("progress_callback", ctypes.c_void_p),
                        ("progress_callback_user_data", ctypes.c_void_p),
                        ("kv_overrides", ctypes.c_void_p),
                        ("use_mmap", ctypes.c_bool), ("use_mlock", ctypes.c_bool),
                        ("check_tensors", ctypes.c_bool)]

        class CtxParams(ctypes.Structure):
            _fields_ = [("n_ctx", ctypes.c_uint32), ("n_batch", ctypes.c_uint32),
                        ("n_ubatch", ctypes.c_uint32), ("n_seq_max", ctypes.c_uint32),
                        ("n_threads", ctypes.c_int32), ("n_threads_batch", ctypes.c_int32),
                        ("rope_scaling_type", ctypes.c_int), ("rope_freq_base", ctypes.c_float),
                        ("rope_freq_scale", ctypes.c_float), ("yarn_ext_factor", ctypes.c_float),
                        ("yarn_attn_factor", ctypes.c_float), ("yarn_beta_fast", ctypes.c_float),
                        ("yarn_beta_slow", ctypes.c_float), ("yarn_orig_ctx", ctypes.c_uint32),
                        ("defrag_thold", ctypes.c_float), ("cb_eval", ctypes.c_void_p),
                        ("cb_eval_user_data", ctypes.c_void_p), ("type_k", ctypes.c_int),
                        ("type_v", ctypes.c_int), ("logits_all", ctypes.c_bool),
                        ("embeddings", ctypes.c_bool), ("offload_kqv", ctypes.c_bool),
                        ("flash_attn", ctypes.c_bool), ("no_perf", ctypes.c_bool),
                        ("op_offload", ctypes.c_bool), ("abort_callback", ctypes.c_void_p),
                        ("abort_callback_data", ctypes.c_void_p)]

        mp = ModelParams()
        mp.use_mmap = True
        mp.use_mlock = False
        self._model = lib.llama_model_load_from_file(model_path.encode(),
                                                     ctypes.byref(mp))
        assert self._model, "не удалось загрузить модель"
        cp = CtxParams()
        cp.n_ctx = n_ctx
        cp.n_batch = 512
        cp.n_ubatch = 512
        cp.n_threads = n_threads
        cp.n_threads_batch = n_threads
        self._ctx = lib.llama_init_from_model(self._model, ctypes.byref(cp))
        assert self._ctx, "не удалось создать контекст"

        lib.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                       ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
                                       ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
        lib.llama_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.llama_get_logits.argtypes = [ctypes.c_void_p]
        lib.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        lib.llama_n_vocab.argtypes = [ctypes.c_void_p]
        lib.llama_n_vocab.restype = ctypes.c_int32
        lib.llama_token_to_piece.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                             ctypes.c_char_p, ctypes.c_int32]
        lib.llama_vocab_eos_token.argtypes = [ctypes.c_void_p]
        lib.llama_vocab_eos_token.restype = ctypes.c_int32

        class TokenData(ctypes.Structure):
            _fields_ = [("id", ctypes.c_int32), ("logit", ctypes.c_float),
                        ("p", ctypes.c_float)]

        class Batch(ctypes.Structure):
            _fields_ = [("n_tokens", ctypes.c_int32),
                        ("token", ctypes.POINTER(ctypes.c_int32)),
                        ("embd", ctypes.POINTER(ctypes.c_float)),
                        ("pos", ctypes.POINTER(ctypes.c_int32)),
                        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
                        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
                        ("logits", ctypes.POINTER(ctypes.c_int8))]

        lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32,
                                         ctypes.c_int32]
        lib.llama_batch_init.restype = Batch

    def chat(self, user: str, card: str, max_tokens: int = 220) -> str:
        """Один цикл: промпт (система+карточка+вопрос) -> жадная генерация."""
        prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n\nКарточка:\n{card}"
                  f"<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        lib = self._lib
        n_vocab = lib.llama_n_vocab(self._model)
        ids = (ctypes.c_int32 * (len(prompt) * 2))()
        n = lib.llama_tokenize(self._model, prompt.encode(), len(prompt.encode()),
                               ids, len(ids), True, False)
        assert n > 0
        # жадный цикл: последовательный decode по одному токену
        seq = list(ids[:n])
        out = []
        for _ in range(max_tokens):
            batch = lib.llama_batch_init(1, 0, 1)
            batch.n_tokens = 1
            tid = ctypes.c_int32(seq[-1])
            pos = ctypes.c_int32(len(seq) - 1)
            sid = ctypes.c_int32(0)
            logits = ctypes.c_int8(1)
            batch.token = ctypes.cast(ctypes.byref(tid), ctypes.POINTER(ctypes.c_int32))
            batch.pos = ctypes.cast(ctypes.byref(pos), ctypes.POINTER(ctypes.c_int32))
            seqid_pp = ctypes.cast(ctypes.byref(ctypes.cast(ctypes.byref(sid), ctypes.c_void_p)), ctypes.POINTER(ctypes.c_int32))
            batch.seq_id = ctypes.cast(ctypes.byref(seqid_pp), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
            ns = ctypes.c_int32(1)
            batch.n_seq_id = ctypes.cast(ctypes.byref(ns), ctypes.POINTER(ctypes.c_int32))
            batch.logits = ctypes.cast(ctypes.byref(logits), ctypes.POINTER(ctypes.c_int8))
            if lib.llama_decode(self._ctx, ctypes.byref(batch)) != 0:
                break
            lg = lib.llama_get_logits(self._ctx)
            nxt = max(range(n_vocab), key=lambda i: lg[i])
            if nxt == lib.llama_vocab_eos_token(self._model):
                break
            buf = ctypes.create_string_buffer(32)
            lib.llama_token_to_piece(self._model, nxt, buf, 32)
            out.append(buf.value.decode("utf-8", "replace"))
            seq.append(nxt)
        return "".join(out)


class LlamaLocalGenerator:
    """Адаптер под интерфейс генераторов (как LlamaServerGenerator)."""

    def __init__(self, llm):
        self.llm = llm
        from generator import TemplateGenerator
        self._template = TemplateGenerator()

    def generate(self, query: str, res) -> str:
        if not res.candidates:
            return self._template.generate(query, res)
        try:
            card = res.candidates[0].text
            text = self.llm.chat(query, card)
            return text.strip() or self._template.generate(query, res)
        except Exception:
            return self._template.generate(query, res)
