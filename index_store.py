"""Сегментный индекс на диске (этап E1 плана).

Формат (папка состояния):
  manifest.json          — {version, segments: [{name, n}], deleted: [gid...]}
                           пишется атомарно (tmp + rename): падение посреди
                           записи не роняет индекс
  seg_NNNNNN/meta.json   — {n, sources}
  seg_NNNNNN/vecs.npz    — float16 (n, d) эмбеддинги
  seg_NNNNNN/docs.zlib   — zlib(pickle): texts, metas, sims

Глобальные структуры в RAM (перестраиваются на load/commit):
  terms  — отсортированные термины, df, ptr/postings (CSR, int32)
           поиск термина — бинарный поиск, без dict-of-dict
  deleted — tombstones (bool по глобальному docid)

Документоориентированные списки (texts/metas/sims) — зеркало MemoryTable;
источник истины для save/load — сегменты, MemoryTable восстанавливается из них.
"""
import bisect
import json
import os
import pickle
import zlib
from collections import Counter

import numpy as np

from memory import simhash64, tokenize
from textnorm import (edit_dist, fix_layout_token, norm_terms,
                      spell_deletes, stem)

BM25_K1 = 1.2
BM25_B = 0.75


class IndexStore:
    def __init__(self):
        self.texts: list[str] = []
        self.metas: list[dict | None] = []
        self.sims: list[int] = []
        self.vecs: np.ndarray | None = None      # float16 (n, d) конкатенация
        # CSR-постинги
        self.terms: list[str] = []
        self.df: np.ndarray | None = None
        self.ptr: np.ndarray | None = None
        self.pdocs: np.ndarray | None = None
        self.ptf: np.ndarray | None = None
        self.dl: np.ndarray | None = None
        self.avgdl: float = 1.0
        self.deleted: np.ndarray | None = None
        self.d_model: int = 1
        self._dirty = True
        self._vocab = None
        self._spell = None
        self._surface = None
        self._ivf = None
        self._ivf_n = -1
        self.ann_min_docs = 20000   # ниже порога — точный перебор
        self._committed_n = 0

    # ---------- наполнение ----------

    def __len__(self) -> int:
        return len(self.texts)

    def add_docs(self, texts, metas, vecs_f32) -> None:
        """Добавить пачку документов (до commit — «живой» сегмент)."""
        base = len(self.texts)
        for t, m in zip(texts, metas):
            self.texts.append(t)
            self.metas.append(m)
            self.sims.append(simhash64(t))
        v = np.asarray(vecs_f32, dtype=np.float16)
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
        self.d_model = self.vecs.shape[1] if self.vecs is not None else self.d_model
        if self.deleted is None:
            self.deleted = np.zeros(len(self.texts), dtype=bool)
        elif len(self.deleted) < len(self.texts):
            self.deleted = np.concatenate(
                [self.deleted, np.zeros(len(self.texts) - len(self.deleted), dtype=bool)])
        self._dirty = True

    def finalize(self) -> None:
        """Перестроить CSR после пакетного добавления + ANN при масштабе."""
        self._rebuild()
        if len(self.texts) >= self.ann_min_docs:
            if self._ivf is None or self._ivf_n != len(self.texts):
                from ann import IVF
                self._ivf = IVF().fit(self.vecs)
                self._ivf_n = len(self.texts)
        else:
            self._ivf = None

    def search_dense(self, vec, k: int = 50, source: str | None = None):
        """Топ-k по косинусу; IVF при >= ann_min_docs, иначе точный перебор."""
        if self._dirty:
            self._rebuild()
        if self._ivf is not None and source is None:
            hits = self._ivf.query(vec, k)
            # адаптивность: слабое совпадение -> ширим поиск (точность)
            if len(hits) < k or hits[0][1] < 0.35:
                hits = self._ivf.query(vec, k, nprobe=48)
        else:
            sims = self.vecs.astype(np.float32) @ vec
            kk = min(k, len(self.texts))
            top = np.argpartition(-sims, kk - 1)[:kk]
            hits = [(int(i), float(sims[i])) for i in top[np.argsort(-sims[top])]]
        if source is not None:
            hits = [(i, s) for i, s in hits
                    if source in str((self.metas[i] or {}).get("source", ""))]
        return hits[:k]

    def _rebuild(self) -> None:
        n = len(self.texts)
        pos_lists: dict[str, list] = {}
        dl = np.zeros(n, dtype=np.int32)
        for i, t in enumerate(self.texts):
            toks = norm_terms(t)
            dl[i] = len(toks)
            per: dict[str, list] = {}
            for p, tok in enumerate(toks):
                per.setdefault(tok, []).append(p)
            for term, pl in per.items():
                pos_lists.setdefault(term, []).append((i, pl))
        self.terms = sorted(pos_lists)
        # поверхностные формы (не стеммы) — для честного symspell:
        # «рудн» легален через «рудник», а токен «рудно» — нет
        from memory import tokenize as _tok
        self._surface = set()
        for t in self.texts:
            self._surface.update(_tok(t))
        m = sum(len(v) for v in pos_lists.values())
        npos = sum(len(pl) for v in pos_lists.values() for _, pl in v)
        ptr = np.zeros(len(self.terms) + 1, dtype=np.int64)
        pptr = np.zeros(len(self.terms) + 1, dtype=np.int64)
        pdocs = np.zeros(m, dtype=np.int32)
        ptf = np.zeros(m, dtype=np.int32)
        ppos = np.zeros(npos, dtype=np.int32)
        df = np.zeros(len(self.terms), dtype=np.int32)
        pos = off = 0
        for j, term in enumerate(self.terms):
            lst = pos_lists[term]
            ptr[j] = pos
            pptr[j] = off
            for k, (d, pl) in enumerate(lst):
                pdocs[pos + k] = d
                ptf[pos + k] = len(pl)
                ppos[off:off + len(pl)] = pl
                off += len(pl)
            pos += len(lst)
            df[j] = len(lst)
        ptr[len(self.terms)] = pos
        pptr[len(self.terms)] = off
        self.df, self.ptr, self.pdocs, self.ptf = df, ptr, pdocs, ptf
        self.pptr = pptr
        self.ppos = ppos
        self.dl = dl
        self.avgdl = max(1.0, float(dl.mean()) if n else 1.0)
        if self.deleted is None or len(self.deleted) != n:
            old = self.deleted if self.deleted is not None else np.zeros(0, bool)
            self.deleted = np.concatenate([old, np.zeros(n - len(old), dtype=bool)])
        self._vocab = set(self.terms)
        self._spell = None
        self._dirty = False

    # ---------- поиск ----------

    def _tid(self, term: str) -> int | None:
        i = bisect.bisect_left(self.terms, term)
        return i if i < len(self.terms) and self.terms[i] == term else None

    def query_terms(self, query: str) -> list:
        """Термины запроса: раскладка -> symspell (только df=0) -> стемминг."""
        if self._dirty:
            self._rebuild()
        vocab = self._vocab
        out = []
        for tok in tokenize(query):
            tok = fix_layout_token(tok, vocab)
            st = stem(tok)
            if (vocab is not None and self._surface is not None
                    and tok not in self._surface and 4 <= len(tok) <= 28):
                # формы нет в корпусе -> вероятная опечатка (EVALS огр. 2)
                fixed = self._spell_correct(tok)
                if fixed:
                    st = stem(fixed)
            out.append(st)
        return out

    def _spell_correct(self, tok: str):
        """Symspell: общие deletes (hash-ключи), верификация edit_dist <= 2."""
        if self._spell is None:
            self._spell = {}
            for tid, term in enumerate(self.terms):
                if self.df[tid] < 2 or len(term) < 4:
                    continue
                for d in spell_deletes(term):
                    self._spell.setdefault(hash(d), []).append(tid)
        best, best_key = None, (3, 0)
        for d in spell_deletes(tok):
            for tid in self._spell.get(hash(d), ()):
                term = self.terms[tid]
                if abs(len(term) - len(tok)) > 2:
                    continue
                ed = edit_dist(tok, term)
                if ed <= 2:
                    key = (ed, -int(self.df[tid]))
                    if key < best_key:
                        best_key, best = key, term
        return best

    def phrase_docs(self, phrase_terms):
        """Docids, где термины фразы идут подряд (позиционное пересечение)."""
        if self._dirty:
            self._rebuild()
        if not phrase_terms:
            return set()
        dps = []
        for t in phrase_terms:
            tid = self._tid(t)
            if tid is None:
                return set()
            s = slice(self.ptr[tid], self.ptr[tid + 1])
            sp = slice(self.pptr[tid], self.pptr[tid + 1])
            docs, tf, pos = self.pdocs[s], self.ptf[s], self.ppos[sp]
            d = {}
            off = 0
            for doc, c in zip(docs, tf):
                d[int(doc)] = pos[off:off + c]
                off += c
            dps.append(d)
        if len(dps) == 1:
            return set(dps[0])
        # кандидаты — из редчайшего термина (дёшево), но смежность
        # проверяем строго в порядке терминов фразы
        rare = min(range(len(dps)), key=lambda i: len(dps[i]))
        cand = set(dps[rare])
        cur = {d: set(ps) for d, ps in dps[0].items() if d in cand}
        for i in range(len(dps) - 1):
            nxt = {}
            for doc, pa in cur.items():
                if doc not in dps[i + 1]:
                    continue
                shifted = {int(p) + 1 for p in pa}
                if shifted & set(int(p) for p in dps[i + 1][doc]):
                    nxt[doc] = set(dps[i + 1][doc])
            cur = nxt
            if not cur:
                return set()
        return set(cur)

    def search_bm25(self, query, k: int = 50, weights: dict | None = None,
                    phrases: list | None = None):
        """Топ-k по BM25; query — сырая строка (или список терминов).
        weights: вес термина (экспансия RM3 идёт с пониженным весом)."""
        if self._dirty:
            self._rebuild()
        if isinstance(query, str):
            query_terms = self.query_terms(query)
        else:
            query_terms = query
        n = len(self.texts)
        if n == 0:
            return []
        scores = np.zeros(n, dtype=np.float32)
        for term in set(query_terms):
            tid = self._tid(term)
            if tid is None:
                continue
            wq = weights.get(term, 1.0) if weights else 1.0
            s = slice(self.ptr[tid], self.ptr[tid + 1])
            docs, tf = self.pdocs[s], self.ptf[s]
            dfi = int(self.df[tid])
            idf = float(np.log1p((n - dfi + 0.5) / (dfi + 0.5)))
            norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.dl[docs] / self.avgdl)
            np.add.at(scores, docs, wq * idf * tf * (BM25_K1 + 1) / norm)
        if phrases:
            for ph in phrases:
                docs = self.phrase_docs(ph)
                if not docs:
                    continue
                idf_sum = 0.0
                for t in set(ph):
                    tid = self._tid(t)
                    if tid is not None:
                        dfi = int(self.df[tid])
                        idf_sum += float(np.log1p((n - dfi + 0.5) / (dfi + 0.5)))
                np.add.at(scores, list(docs), 2.0 * idf_sum)
        scores[self.deleted] = 0.0
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[scores[top] > 0]
        return [(int(i), float(scores[i])) for i in top[np.argsort(-scores[top])]]

    def remove_source(self, substr: str) -> int:
        """Tombstone всех документов, чей source содержит substr."""
        cnt = 0
        for i, m in enumerate(self.metas):
            if not self.deleted[i] and m and substr in str(m.get("source", "")):
                self.deleted[i] = True
                cnt += 1
        return cnt

    # ---------- persistence ----------

    def commit(self, path: str) -> None:
        """Дописать «живые» документы новым сегментом + атомарный манифест."""
        os.makedirs(path, exist_ok=True)
        new = len(self.texts) - self._committed_n
        manifest = self._read_manifest(path)
        if new > 0:
            name = "seg_%06d" % len(manifest["segments"])
            seg_dir = os.path.join(path, name)
            os.makedirs(seg_dir, exist_ok=True)
            sl = slice(self._committed_n, len(self.texts))
            texts = self.texts[sl]
            metas = self.metas[sl]
            sims = self.sims[sl]
            vecs = self.vecs[sl]
            with open(os.path.join(seg_dir, "docs.zlib"), "wb") as f:
                f.write(zlib.compress(
                    pickle.dumps({"texts": texts, "metas": metas, "sims": sims},
                                 protocol=4), 6))
            np.savez_compressed(os.path.join(seg_dir, "vecs.npz"), vecs=vecs)
            with open(os.path.join(seg_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"n": new,
                           "sources": sorted({str((m or {}).get("source", ""))
                                              for m in metas})[:50]}, f,
                          ensure_ascii=False)
            manifest["segments"].append({"name": name, "n": new})
            self._committed_n = len(self.texts)
        manifest["deleted"] = [int(i) for i in np.flatnonzero(self.deleted)]
        self._write_manifest(path, manifest)

    @classmethod
    def open(cls, path: str) -> "IndexStore":
        store = cls()
        manifest = cls._read_manifest(path)
        for seg in manifest["segments"]:
            seg_dir = os.path.join(path, seg["name"])
            with open(os.path.join(seg_dir, "docs.zlib"), "rb") as f:
                blob = pickle.loads(zlib.decompress(f.read()))
            store.texts.extend(blob["texts"])
            store.metas.extend(blob["metas"])
            store.sims.extend(blob["sims"])
            v = np.load(os.path.join(seg_dir, "vecs.npz"))["vecs"]
            store.vecs = v if store.vecs is None else np.vstack([store.vecs, v])
        store.d_model = int(store.vecs.shape[1]) if store.vecs is not None else 1
        store._committed_n = len(store.texts)
        store.deleted = np.zeros(len(store.texts), dtype=bool)
        for gid in manifest.get("deleted", []):
            if 0 <= gid < len(store.deleted):
                store.deleted[gid] = True
        store._dirty = True
        store.finalize()
        return store

    @staticmethod
    def _read_manifest(path: str) -> dict:
        mf = os.path.join(path, "manifest.json")
        if not os.path.exists(mf):
            return {"version": 1, "segments": [], "deleted": []}
        with open(mf, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_manifest(path: str, manifest: dict) -> None:
        tmp = os.path.join(path, "manifest.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(path, "manifest.json"))
