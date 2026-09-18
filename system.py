"""Оркестрация: эмбеддер + мозг + таблица памяти + калибровка + DAN.

Ответ мозга всегда несет три числа уверенности:
  familiarity — знакомость темы (cos восстановленного вектора с запросом);
  conf        — калиброванная уверенность по топ-1 кандидату;
  margin      — запас топ-1 над топ-2 (неоднозначность).

Зоны: unknown (не помню) | unsure (кажется) | ambiguous (два кандидата) | confident.
"""
import json
import os

import numpy as np

from calibration import apply_calib
from config import FlyConfig
from flybrain import FlyBrain
from index_store import IndexStore
from memory import Candidate, MemoryTable, RecallResult


class FlySystem:
    def __init__(self, cfg: FlyConfig | None = None, embedder=None,
                 gen_backend: str = "template", ollama_model: str = "qwen2.5:0.5b",
                 llama_host: str = "http://127.0.0.1:8081"):
        self.cfg = cfg or FlyConfig()
        if embedder is not None:
            self.embedder = embedder
            self.cfg.d_model = embedder.d_model
        else:
            from embedder import HashingEmbedder
            self.embedder = HashingEmbedder(self.cfg.d_model)
        self.brain = FlyBrain(self.cfg)
        self.memory = MemoryTable()
        self.last_result: RecallResult | None = None
        self.last_query: str | None = None
        from generator import make_generator
        self.generator = make_generator(gen_backend, ollama_model,
                                        llama_host=llama_host)
        from index_store import IndexStore
        from ranker import LinearRanker
        self.store = IndexStore()
        self.ranker = LinearRanker()
        self.journal: list[dict] = []
        self.doc_feedback: dict[int, float] = {}
        self.corrections: dict[str, list[int]] = {}
        self.session: list[str] = []   # последние запросы (контекст местоимений)
        self._image_embedder = None
        self.species_info: dict[str, dict] = {}
        self._img_hashes: set[str] = set()   # дедуп фото при переиндексации
        self._load_calib()

    def _load_calib(self) -> None:
        """calib.json рядом с запуском: theta_low, calib_a/b, calib_a2/b2 (BM25)."""
        if os.path.exists("calib.json"):
            with open("calib.json", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if hasattr(self.cfg, k) and isinstance(v, (int, float)):
                        setattr(self.cfg, k, float(v))

    # ---------- универсальный поиск ----------

    def _reembed_all(self) -> None:
        """Перевести уже хранимые векторы в новое пространство эмбеддера.

        Сценарий: memorize()-факты записались до fit() (uniform-веса),
        затем индексация корпуса подстроила SIF/PCA — старые векторы
        остались бы в ином пространстве (EVALS, ограничение 3).
        Следы в мозге (W_kc_mbon) освежаем повторным learn на новом векторе.
        """
        if not len(self.store.texts):
            return
        new_vecs = [self.embedder.embed(t) for t in self.store.texts]
        self.store.vecs = np.stack(new_vecs).astype(np.float16)
        self.store._ivf = None  # перестроится при следующем finalize
        for i, v in enumerate(new_vecs):
            self.memory.vecs[i] = v
            # обновляем и след в мозге: повторный learn перевешивает старый
            self.brain.learn(np.asarray(v, dtype=np.float32), reward=+1,
                             weight=0.5)
        self.memory._mat = None

    def index_path(self, root: str, size: int = 2, overlap: int = 1) -> dict:
        """Проиндексировать файл или папку: адаптеры -> нарезка -> дедуп -> one-shot."""
        from loader import chunk_doc
        from sources import load_path
        docs = load_path(root)
        added = dups = 0
        # фаза 1: собрать чанки (без эмбеддинга)
        pending: list[tuple[str, dict]] = []
        for d in docs:
            for text, heading in chunk_doc(d["text"], size, overlap):
                if not text or self.memory.has_text(text):
                    dups += int(bool(text))
                    continue
                meta = dict(d["meta"])
                if heading:
                    meta["heading"] = heading
                pending.append((text, meta))
        # фаза 2: SIF/PCA-подстройка эмбеддера под корпус, затем эмбеддинг
        if len(pending) >= 50 and hasattr(self.embedder, "fit"):
            n_prior = len(self.store.texts)
            self.embedder.fit([t for t, _ in pending])
            if n_prior:
                self._reembed_all()
        texts, metas, vecs = [], [], []
        for text, meta in pending:
            x = self.embedder.embed(text)
            self.memory.store(text, x, meta=meta)
            self.brain.learn(x, reward=+1)
            texts.append(text)
            metas.append(meta)
            vecs.append(x)
            added += 1
        if vecs:
            self.store.add_docs(texts, metas, np.stack(vecs).astype(np.float32))
            self.store.finalize()
        return {"files": len({d["meta"]["source"] for d in docs}),
                "added": added, "dups": dups}

    def _ensure_sparse(self):
        if self.store._dirty:
            self.store.finalize()

    def search(self, query: str, k: int | None = None, source: str | None = None):
        """Гибридный поиск: dense top-50 + BM25 top-50 -> RRF -> зоны уверенности.

        Возвращает RecallResult; у кандидатов conf = калиброванный dense-косинус,
        если эпизод не попал в dense-список — нейтральные 0.5 (sparse-находка).
        """
        from search import rrf_fuse
        from memory import tokenize
        from calibration import sigmoid as _sigmoid
        cfg = self.cfg
        k = k or cfg.topk_mem
        import re as _re
        # операторы: source:<подстрока> heading:<подстрока> -<исключаемое>
        op_src = op_head = None
        excl_words = []
        q_clean = []
        for tok in query.split():
            low = tok.lower()
            if low.startswith("source:") and len(tok) > 7:
                op_src = tok[7:]
            elif low.startswith("heading:") and len(tok) > 8:
                op_head = tok[8:]
            elif tok.startswith("-") and len(tok) > 1:
                excl_words.append(tok[1:])
            else:
                q_clean.append(tok)
        query = " ".join(q_clean)
        source = op_src or source
        phrases_raw = _re.findall(r'"([^"]+)"', query)
        q_wo = _re.sub(r'"[^"]*"', " ", query) if phrases_raw else query
        # документы с исключаемыми терминами (по постингам, стемминг учтён)
        from textnorm import norm_terms as _nt
        excluded_ids = set()
        for w in excl_words:
            for t in set(_nt(w)):
                tid = self.store._tid(t)
                if tid is not None:
                    excluded_ids |= set(
                        self.store.pdocs[self.store.ptr[tid]:self.store.ptr[tid + 1]].tolist())
        op_head_l = op_head.lower() if op_head else None
        phrases = [self.store.query_terms(p) for p in phrases_raw]
        x = self.embedder.embed(q_wo)
        if self.session and q_wo.strip():
            # сессионный контекст: «он», «это» и т.п. разрешаются через
            # смешение с векторами недавних запросов (п.29 плана)
            ctx = np.stack([self.embedder.embed(sq) for sq in self.session[-3:]])
            blend = cfg.session_w * ctx.mean(axis=0) + (1.0 - cfg.session_w) * x
            nb = float(np.linalg.norm(blend))
            if nb > 0:
                x = (blend / nb).astype(np.float32)
        fam = self.brain.familiarity(x)
        # запрос целиком в кавычках: dense на пустом тексте дал бы
        # произвольный порядок и затмил бы фразовый бонус
        dense_ids = ([i for i, _ in self.store.search_dense(x, 50, source=source)]
                     if q_wo.strip() else [])
        self._ensure_sparse()
        bm_hits = self.store.search_bm25(q_wo, 50, phrases=phrases)
        if source is not None:
            bm_hits = [(i, s) for i, s in bm_hits
                       if source in str((self.memory.meta[i] or {}).get("source", ""))]
        sparse_lists = [[i for i, _ in bm_hits]]
        from answer_extract import is_question
        if cfg.rm3_enabled and is_question(q_wo) and len(bm_hits) >= 3:
            from search import rm3_expand
            from textnorm import norm_terms
            qt0 = self.store.query_terms(query)
            exp = set(rm3_expand(norm_terms, qt0,
                                 [self.memory.texts[i] for i, _ in bm_hits[:5]],
                                 cfg.rm3_n))
            if exp:
                all_terms = list(set(qt0) | exp)
                weights = {t: (cfg.rm3_w if t in exp else 1.0) for t in all_terms}
                hits2 = self.store.search_bm25(all_terms, 50, weights=weights)
                if source is not None:
                    hits2 = [(i, s) for i, s in hits2
                             if source in str((self.memory.meta[i] or {}).get("source", ""))]
                if hits2:
                    sparse_lists.append([i for i, _ in hits2])
        qt = self.store.query_terms(q_wo)
        # сессионный контекст в sparse-канале: редкие термины последнего
        # запроса сессии добавляются с пониженным весом (иначе бленд
        # dense-вектора слабо перебивает BM25-паритет)
        if self.session and q_wo.strip():
            ctx_add = []
            for t in self.store.query_terms(self.session[-1]):
                tid = self.store._tid(t)
                if (tid is not None and self.store.df[tid] >= 2
                        and t not in set(qt)):
                    ctx_add.append(t)
            if ctx_add:
                hits_ctx = self.store.search_bm25(
                    ctx_add, 30, weights={t: cfg.session_w for t in ctx_add})
                if hits_ctx:
                    sparse_lists.append([i for i, _ in hits_ctx])
        sparse_ids = [i for lst in sparse_lists for i in lst]
        bm_max = max((s for _, s in bm_hits), default=0.0)
        bm_norm = {i: (s / bm_max if bm_max > 0 else 0.0) for i, s in bm_hits}
        raw_by_id = {i: s for i, s in self.store.search_dense(x, 50, source=source)}
        sparse_set = set(sparse_ids)
        bm_raw = {i: s for i, s in bm_hits}
        order = rrf_fuse(dense_ids, *sparse_lists)[:cfg.rerank_pool]
        pool = []
        for idx in order:
            if not self.memory.active[idx] or self.store.deleted[idx]:
                continue
            if idx in excluded_ids:
                continue
            if op_head_l:
                head = str((self.memory.meta[idx] or {}).get("heading", "")).lower()
                if op_head_l not in head:
                    continue
            raw = raw_by_id.get(idx)
            if raw is not None:
                conf = float(apply_calib(raw, cfg.calib_a, cfg.calib_b))
                if idx in sparse_set:
                    conf = max(conf, 0.5)  # подтверждено обоими ретриверами
            else:
                # только BM25: уверенность из нормированного BM25-скора
                conf = float(_sigmoid(cfg.calib_a2 * bm_norm.get(idx, 0.0) + cfg.calib_b2))
            pool.append(Candidate(idx, self.memory.texts[idx],
                                  raw if raw is not None else 0.0, conf))
        # LTR-переранжирование пула
        feats = {c.idx: self._features_from(qt, c.idx, x, c.raw_cos,
                                            bm_raw.get(c.idx, 0.0)) for c in pool}
        pool.sort(key=lambda c: self.ranker.score(feats[c.idx]), reverse=True)
        # детерминированные коррекции: документ, исправленный пользователем
        # для этого запроса, поднимается (LTR v2 — закрывает конфликт dan)
        sig = " ".join(sorted(set(qt)))
        if sig in self.corrections:
            bonus = {idx: 2.0 - 0.5 * p for p, idx in
                     enumerate(self.corrections[sig][:4])}
            pool.sort(key=lambda c: self.ranker.score(feats[c.idx])
                      + bonus.get(c.idx, 0.0), reverse=True)
        cands = pool[:k]
        margin = (cands[0].raw_cos - cands[1].raw_cos) if len(cands) >= 2 else 1.0
        res = RecallResult(familiarity=fam, zone=self._zone(fam, cands, margin),
                           candidates=cands, margin=margin)
        self.last_result = res
        self.last_query = query
        if q_wo.strip():
            self.session.append(q_wo)
            del self.session[:-self.cfg.session_n]
        return res

    def citation(self, cand) -> str:
        """Цитата для ответа: источник + заголовок/строка/страница, если есть."""
        m = self.memory.meta[cand.idx] or {}
        loc = m.get("source", "?")
        for key in ("heading", "page", "row", "line"):
            if m.get(key) is not None:
                loc += f" · {key} {m[key]}"
        return loc

    # ---------- API ----------

    def memorize(self, text: str, reward: float = +1.0) -> dict:
        """One-shot запоминание: эпизод сразу в таблице и в W_kc_mbon."""
        x = self.embedder.embed(text)
        idx = self.memory.store(text, x)
        self.store.add_docs([text], [self.memory.meta[idx]], np.stack([x]))
        stats = self.brain.learn(x, reward=reward)
        stats["memory_idx"] = idx
        return stats

    def ask(self, query: str, k: int | None = None) -> RecallResult:
        """Узнавание: знакомость + top-k кандидатов с уверенностью и зоной ответа."""
        cfg = self.cfg
        k = k or cfg.topk_mem
        x = self.embedder.embed(query)
        fam = self.brain.familiarity(x)
        cands = []
        for idx, raw in self.memory.topk(x, k * 2):  # запас, т.к. часть может быть неактивна
            if not self.memory.active[idx]:
                continue
            conf = float(apply_calib(raw, cfg.calib_a, cfg.calib_b))
            cands.append(Candidate(idx, self.memory.texts[idx], raw, conf))
            if len(cands) >= k:
                break
        margin = (cands[0].raw_cos - cands[1].raw_cos) if len(cands) >= 2 else 1.0
        res = RecallResult(familiarity=fam, zone=self._zone(fam, cands, margin),
                           candidates=cands, margin=margin)
        self.last_result = res
        self.last_query = query
        return res

    def answer(self, query: str, k: int | None = None):
        """Полный цикл: узнавание -> генерация -> extractive QA.
        Возвращает (result, text)."""
        res = self.ask(query, k)
        text = self.generator.generate(query, res)
        if res.candidates and res.zone in ("confident", "unsure"):
            from answer_extract import extract_answer, is_question
            if is_question(query):
                ext = extract_answer(query, [(c.text, self.memory.meta[c.idx])
                                             for c in res.candidates[:3]])
                if ext and ext["conf"] >= 0.5:
                    if ext["type"] in ("reason", "thing"):
                        text = ext["answer"]
                    else:
                        text = f"{ext['answer']}.  —  «{ext['evidence'][:140]}»"
        return res, text

    def _nbr_dense(self, idx: int, query_vec: np.ndarray) -> float:
        """Макс. косинус с соседними чанками того же источника (+-1)."""
        if self.store.vecs is None:
            return 0.0
        src = (self.memory.meta[idx] or {}).get("source")
        best = 0.0
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(self.memory) and \
                    (self.memory.meta[j] or {}).get("source") == src:
                v = self.store.vecs[j].astype(np.float32)
                best = max(best, float(np.dot(query_vec, v)))
        return best

    def _features_from(self, query_terms: list, idx: int,
                       query_vec: np.ndarray, dense_raw: float,
                       bm_raw: float) -> np.ndarray:
        from textnorm import norm_terms
        from ranker import extract_features
        meta = self.memory.meta[idx] or {}
        text = self.memory.texts[idx]
        doc_terms = set(norm_terms(text))
        rare = set()
        for t in set(query_terms):
            tid = self.store._tid(t)
            if tid is not None and self.store.df[tid] == 1:
                rare.add(t)
        dl = int(self.store.dl[idx]) if self.store.dl is not None else len(doc_terms)
        return extract_features(query_terms, bm_raw, dense_raw, text,
                                meta.get("heading"), doc_terms, rare, dl,
                                self.store.avgdl,
                                self._nbr_dense(idx, query_vec),
                                self.doc_feedback.get(idx, 0.0))

    def _online_learn(self, ev: dict) -> None:
        """Один фидбек -> пары (picked > остальные показанные) -> SGD-шаг."""
        if not ev["shown"]:
            return
        x = self.embedder.embed(ev["q"])
        bm = dict(self.store.search_bm25(ev["q"], 50))
        dense = dict(self.memory.topk(x, 50))
        qt = self.store.query_terms(ev["q"])
        def f(idx):
            return self._features_from(qt, idx, x, dense.get(idx, 0.0),
                                       bm.get(idx, 0.0))
        fw = f(ev["picked"])
        pairs = [(fw, f(i)) for i in ev["shown"]
                 if i != ev["picked"] and i < len(self.memory)]
        if pairs:
            self.ranker.fit_pairs(pairs)

    def feedback(self, sign: float, cand: Candidate | None = None,
                 text: str | None = None) -> dict | None:
        """DAN-канал: sign=+1 усиливает ассоциацию, -1 стирает.

        Наказание взвешивается уверенностью показанного ответа:
        уверенная ошибка стирается сильнее. Эпизод при наказании
        деактивируется в таблице памяти.
        """
        if text is not None:
            idx = self._find(text)
            if idx is None:
                return None
            vec = self.memory.vecs[idx]
        else:
            if cand is None:
                if not self.last_result or not self.last_result.candidates:
                    return None
                cand = self.last_result.top
            idx = cand.idx
            vec = self.memory.vecs[idx]
        weight = 1.0
        if sign < 0 and self.last_result is not None and self.last_result.top is not None:
            weight = 0.5 + self.last_result.top.conf
        stats = self.brain.learn(vec, reward=sign, weight=weight)
        self.memory.set_active(idx, sign > 0)
        self.doc_feedback[idx] = self.doc_feedback.get(idx, 0.0) + sign
        ev = {"q": self.last_query or "",
              "shown": [c.idx for c in (self.last_result.candidates
                                        if self.last_result else [])],
              "picked": idx, "sign": int(sign)}
        self.journal.append(ev)
        self._online_learn(ev)
        sig = " ".join(sorted(set(self.store.query_terms(ev["q"]))))
        lst = self.corrections.setdefault(sig, [])
        if sign > 0:
            if idx in lst:
                lst.remove(idx)
            lst.insert(0, idx)
        elif idx in lst:
            lst.remove(idx)
        return stats

    def _img_embedder(self):
        """None — если ни один бэкенд недоступен (APK v1 без onnxruntime)."""
        if self._image_embedder is None:
            try:
                from image_embedder import ImageEmbedder
                self._image_embedder = ImageEmbedder()
            except RuntimeError:
                self._image_embedder = False
        return self._image_embedder or None

    def index_species(self, root: str) -> dict:
        """База видов: root/<Вид>/card.md (текст) + *.jpg/png (экземпляры).

        Фото эмбеддятся ImageEmbedder'ом в то же 512-d пространство;
        карточки нарезаются как обычный текст. meta: species, kind.
        """
        import glob as _glob
        import re as _re
        from loader import chunk_doc
        added_img = added_card = 0
        texts, metas, vecs = [], [], []
        for sp_dir in sorted(_glob.glob(os.path.join(root, "*/"))):
            species = os.path.basename(sp_dir.rstrip("/"))
            card = os.path.join(sp_dir, "card.md")
            if os.path.exists(card):
                text = open(card, encoding="utf-8").read()
                m = _re.search(r"ядовит\w*[:\s]*(да|нет)", text.lower())
                # трёхзначность: неизвестное = опасное (каркасы WFO)
                self.species_info[species] = {
                    "toxic": (True if m and m.group(1) == "да"
                              else (False if m else None))}
                meta0 = {"source": card, "species": species, "kind": "card"}
                for ch, heading in chunk_doc(text, 2, 1):
                    if not ch or self.memory.has_text(ch):
                        continue
                    x = self.embedder.embed(ch)
                    meta = dict(meta0)
                    if heading:
                        meta["heading"] = heading
                    texts.append(ch)
                    metas.append(meta)
                    vecs.append(x)
                    self.memory.store(ch, x, meta=meta)
                    self.brain.learn(x, reward=+1)
                    added_card += 1
            imgs = sorted(_glob.glob(sp_dir + "*.jpg") + _glob.glob(sp_dir + "*.jpeg")
                          + _glob.glob(sp_dir + "*.png"))
            # дедуп: одинаковые файлы (копии, повторная индексация) пропускаем
            import hashlib as _hl
            uniq = []
            for p in imgs:
                h = _hl.md5(open(p, "rb").read()).hexdigest()
                if h not in self._img_hashes:
                    self._img_hashes.add(h)
                    uniq.append(p)
            imgs = uniq
            _emb = self._img_embedder()
            if imgs and _emb is None:
                imgs = []  # зрение недоступно: карточки индексируются, фото — нет
            if imgs:
                embs = _emb.embed_batch(imgs)
                for p, x in zip(imgs, embs):
                    meta = {"source": os.path.abspath(p), "species": species,
                            "kind": "image"}
                    texts.append(f"[фото] {species} ({os.path.basename(p)})")
                    metas.append(meta)
                    vecs.append(x)
                    self.memory.store(texts[-1], x, meta=meta)
                    added_img += 1
        if texts:
            # единый add_docs: store docid == memory index (инвариант!)
            self.store.add_docs(texts, metas, np.stack(vecs).astype(np.float32))
        self.store.finalize()
        return {"cards": added_card, "images": added_img,
                "species": len(self.species_info)}

    def recognize(self, image_path: str, k: int = 3,
                  threshold: float | None = None):
        """Фото -> вид -> карточка с советами.

        Порог по умолчанию — cfg.recognize_threshold (калибруется
        calibrate_recognize.py -> recognizer_calib.json).

        БЕЗОПАСНОСТЬ: при cos < threshold вердикт None ("не уверен —
        не употребляй"); для ядовитых видов совет предваряется предупреждением.
        """
        if self._img_embedder() is None:
            return {"query": image_path, "certain": False, "verdict": None,
                    "error": "зрение недоступно в этой сборке (нет onnxruntime)",
                    "matches": []}
        if threshold is None:
            threshold = self.cfg.recognize_threshold
        try:
            q = self._img_embedder().embed_image_tta(image_path)
        except Exception as e:
            return {"query": image_path, "certain": False, "verdict": None,
                    "error": f"зрение временно недоступно: {e}",
                    "matches": []}
        hits = self.store.search_dense(q, 60)
        cands = []
        for idx, cos in hits:
            m = self.memory.meta[idx] or {}
            if m.get("kind") != "image":
                continue
            cands.append((idx, cos, m))
            if len(cands) >= k:
                break
        out = {"query": image_path, "certain": False, "verdict": None,
               "matches": [{"species": m.get("species"), "cos": round(c, 3),
                            "photo": m.get("source")} for _, c, m in cands]}
        # уверенность = порог cos И отрыв от ближайшего ДРУГОГО вида
        # (по fine-grained тестам: у близнецов косинусы плотные ~0.96,
        # решает margin, а не абсолютное значение)
        species_best: dict = {}
        for _, c, m in cands:
            sp = m.get("species")
            species_best[sp] = max(species_best.get(sp, 0.0), c)
        ranked = sorted(species_best.items(), key=lambda kv: -kv[1])
        margin_ok = True
        if len(ranked) >= 2:
            margin_ok = (ranked[0][1] - ranked[1][1]) >= self.cfg.recognize_margin
        if cands and ranked and ranked[0][1] >= threshold and margin_ok:
            species = ranked[0][0]
            out["certain"] = True
            out["verdict"] = species
            info = self.species_info.get(species, {})
            q_text = (f"{species}: чем опасен, как выглядит и как применять "
                      f"в выживании")
            res, text = self.answer(q_text)
            if info.get("toxic") is True:
                text = ("⚠️ ВИД ЯДОВИТ. Не пробовать на вкус. " + text)
            elif info.get("toxic") is None:
                text = ("⚠️ СЪЕДОБНОСТЬ НЕИЗВЕСТНА — не пробовать на вкус. " + text)
            out["advice"] = text
            out["advice_zone"] = res.zone
        return out

    def consolidate(self) -> dict:
        """Консолидация «сна» (P6): офлайн-проход по памяти.

        1. Анти-затухание: эпизоды, которые часто показывались
           (>=2 показа в журнале), но знакомость низкая (<0.3) —
           повторный learn с малым весом.
        2. Деактивация системно наказываемых (doc_feedback <= -2).
        3. PCA/SIF-пересчёт, если корпус вырос >20% с последнего fit.
        """
        report = {"refreshed": 0, "deactivated": 0, "pca_refit": False}
        # 1. часто показанные, но слабые
        shown_count: dict[int, int] = {}
        for ev in self.journal:
            for idx in ev.get("shown", []):
                shown_count[idx] = shown_count.get(idx, 0) + 1
        for idx, cnt in shown_count.items():
            if cnt >= 2 and idx < len(self.memory):
                vec = self.memory.vecs[idx]
                if self.brain.familiarity(vec) < 0.3:
                    self.brain.learn(vec, reward=+1, weight=0.3)
                    report["refreshed"] += 1
        # 2. системные наказанные
        for idx, val in list(self.doc_feedback.items()):
            if val <= -2 and idx < len(self.memory):
                self.memory.set_active(idx, False)
                if idx < len(self.store.deleted):
                    self.store.deleted[idx] = True
                report["deactivated"] += 1
        # 3. PCA-пересчёт при росте корпуса
        emb = self.embedder
        if (hasattr(emb, "_freq") and getattr(emb, "_freq", None)
                and len(self.store) > getattr(emb, "_n_fit", 0) * 1.2):
            emb.fit(list(self.store.texts))
            self._reembed_all()
            report["pca_refit"] = True
        return report

    def new_topic(self) -> None:
        """Сброс сессионного контекста (команда «новая тема»)."""
        self.session.clear()

    def known_facts(self) -> list[str]:
        return [t for t, a in zip(self.memory.texts, self.memory.active) if a]

    # ---------- внутреннее ----------

    def _zone(self, fam: float, cands: list, margin: float) -> str:
        cfg = self.cfg
        # гейт по калиброванной уверенности (P верно | косинус), а не по fam:
        # fam с хэш-эмбеддером разделяет известное/новое лишь слабо
        if not cands or cands[0].conf < cfg.theta_conf:
            return "unknown"
        if cands[0].conf < cfg.theta_mid:
            return "unsure"
        if margin < cfg.delta_margin:
            return "ambiguous"
        return "confident"

    def _find(self, text: str):
        t = text.lower()
        for i, s in enumerate(self.memory.texts):
            if t in s.lower() or s.lower() in t:
                return i
        return None

    # ---------- сохранение / загрузка ----------

    def save(self, path: str) -> None:
        """Состояние = мозг (brain.npz) + сегментный индекс (manifest + seg_*)."""
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(
            os.path.join(path, "brain.npz"),
            W_pn_kc=self.brain.W_pn_kc,
            W_kc_mbon=self.brain.W_kc_mbon,
            scale=self.brain._scale,
        )
        # save = снапшот: старые сегменты удаляются, чтобы не плодить дубликаты
        import shutil
        mf = os.path.join(path, "manifest.json")
        if os.path.exists(mf):
            for seg in IndexStore._read_manifest(path).get("segments", []):
                shutil.rmtree(os.path.join(path, seg["name"]), ignore_errors=True)
            os.remove(mf)
            self.store._committed_n = 0
        self.store.commit(path)
        if hasattr(self.embedder, "state_dict"):
            st = self.embedder.state_dict()
            freq = st["freq"] or {}
            terms = sorted(freq)
            np.savez_compressed(
                os.path.join(path, "embedder.npz"),
                pc=st["pc"] if st["pc"] is not None else np.zeros(0, np.float32),
                terms=np.array(terms, dtype=object),
                counts=np.array([freq[t] for t in terms], dtype=np.int64),
                total=st["total"], n_fit=st["n_fit"])
        np.savez_compressed(os.path.join(path, "ranker.npz"),
                            w=self.ranker.w,
                            pair_count=self.ranker.pair_count,
                            n_pairs=self.ranker.n_pairs)
        with open(os.path.join(path, "journal.jsonl"), "w", encoding="utf-8") as f:
            for ev in self.journal:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        with open(os.path.join(path, "doc_feedback.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.doc_feedback.items()}, f)
        with open(os.path.join(path, "corrections.json"), "w", encoding="utf-8") as f:
            json.dump(self.corrections, f, ensure_ascii=False)
        with open(os.path.join(path, "species_info.json"), "w", encoding="utf-8") as f:
            json.dump(self.species_info, f, ensure_ascii=False)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.cfg.__dict__.items()}, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        b = np.load(os.path.join(path, "brain.npz"))
        self.brain.load_state_dict({"W_pn_kc": b["W_pn_kc"], "W_kc_mbon": b["W_kc_mbon"],
                                    "scale": float(b["scale"])})
        # MemoryTable восстанавливается из сегментов (источник истины)
        self.store = IndexStore.open(path)
        en = os.path.join(path, "embedder.npz")
        if os.path.exists(en) and hasattr(self.embedder, "load_state_dict"):
            e = np.load(en, allow_pickle=True)
            freq = {t: int(c) for t, c in zip(list(e["terms"]), list(e["counts"]))}
            pc = e["pc"] if e["pc"].shape[0] else None
            self.embedder.load_state_dict(
                {"freq": freq, "total": int(e["total"]), "pc": pc,
                 "n_fit": int(e["n_fit"])})
        rn = os.path.join(path, "ranker.npz")
        if os.path.exists(rn):
            r = np.load(rn)
            self.ranker.load_state_dict({"w": r["w"], "pair_count": r["pair_count"],
                                         "n_pairs": int(r["n_pairs"])})
        jn = os.path.join(path, "journal.jsonl")
        if os.path.exists(jn):
            with open(jn, encoding="utf-8") as f:
                self.journal = [json.loads(l) for l in f if l.strip()]
        dn = os.path.join(path, "doc_feedback.json")
        if os.path.exists(dn):
            with open(dn, encoding="utf-8") as f:
                self.doc_feedback = {int(k): float(v)
                                     for k, v in json.load(f).items()}
        cn = os.path.join(path, "corrections.json")
        if os.path.exists(cn):
            with open(cn, encoding="utf-8") as f:
                self.corrections = {k: [int(i) for i in v]
                                    for k, v in json.load(f).items()}
        sin = os.path.join(path, "species_info.json")
        if os.path.exists(sin):
            with open(sin, encoding="utf-8") as f:
                self.species_info = json.load(f)
        self.memory = MemoryTable()
        for i, text in enumerate(self.store.texts):
            self.memory.store(text, self.store.vecs[i].astype(np.float32),
                              meta=self.store.metas[i], sim=self.store.sims[i])
            self.memory.set_active(i, not bool(self.store.deleted[i]))
