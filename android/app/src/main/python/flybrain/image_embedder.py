"""ImageEmbedder: MobileNetV3-small (576-d features) -> 512-d пространство
системы через фиксированное ортонормированное случайное проецирование.

Почему не CLIP: ViT-B/32 ~350 МБ — не влезает в бюджет 400 МБ вместе с БД.
MobileNetV3-small — 10.3 МБ, CPU-инференс ~0.1-0.3 с/кадр. Проекция
576->512 (seeded, QR-ортонормирование) мало искажает косинусы (JL) —
image-image поиск остаётся осмысленным; текст и изображения сравниваются
только опосредованно (фото -> вид -> карточка), что для задачи
«фото -> совет по выживанию» и есть правильный пайплайн.

Веса модели — вне проекта (~/.cache/torch), качаются один раз; при
первом импорте без сети и кэша — RuntimeError с понятным сообщением.
"""
import os

import numpy as np


class ImageEmbedder:
    d_model = 512
    src_dim = 576

    def __init__(self, seed: int = 13):
        self.seed = seed
        self._model = None
        self._proj = self._make_projection(seed)

    @staticmethod
    def _make_projection(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        g = rng.normal(size=(ImageEmbedder.src_dim, ImageEmbedder.d_model))
        q, _ = np.linalg.qr(g)          # ортонормирование -> сохранение метрик
        return q.astype(np.float32)

    def _run_jni(self, arr_nchw: np.ndarray) -> np.ndarray:
        """Инференс через onnxruntime-android Java API (chaquopy-мост).
        Любая ошибка JNI -> RuntimeError (деградация, не краш)."""
        try:
            return self._run_jni_inner(arr_nchw)
        except Exception as e:
            raise RuntimeError(f"JNI-инференс недоступен: {e}")

    def _run_jni_inner(self, arr_nchw: np.ndarray) -> np.ndarray:
        try:
            from jnius import autoclass
        except ImportError:
            from java import autoclass   # встроенный мост Chaquopy
        OnnxTensor = autoclass("ai.onnxruntime.OnnxTensor")
        HashMap = autoclass("java.util.HashMap")
        FloatBuffer = autoclass("java.nio.FloatBuffer")
        flat = arr_nchw.astype(np.float32).ravel().tolist()
        fb = FloatBuffer.allocate(len(flat))
        for v in flat:
            fb.put(v)
        fb.flip()
        t = OnnxTensor.createTensor(self._ort_env, fb, [1, 3, 224, 224])
        inputs = HashMap()
        inputs.put("input", t)
        res = self._ort_sess.run(inputs)
        out = res.get(0).getValue()          # FloatBuffer
        n = out.remaining()
        result = np.empty(n, dtype=np.float32)
        for i in range(n):
            result[i] = out.get(i)           # читаем ВЫХОД, не вход
        return result

    # ---------- препроцессинг (PIL + numpy, без torch) ----------
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img) -> np.ndarray:
        """resize(min=256) -> center-crop 224 -> normalize -> NCHW."""
        from PIL import Image
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        w, h = img.size
        scale = 256.0 / min(w, h)
        img = img.resize((max(224, int(w * scale)), max(224, int(h * scale))),
                         Image.BILINEAR)
        nw, nh = img.size
        left, top = max(0, (nw - 224) // 2), max(0, (nh - 224) // 2)
        img = img.crop((left, top, left + 224, top + 224))
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - self._MEAN) / self._STD
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None])

    # ---------- модель (ленивая загрузка) ----------

    def _load(self):
        if self._model is not None:
            return
        # приоритет 1: ONNX (мобильный путь: onnxruntime, 3.7 МБ, без torch)
        here = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(here, "models", "mobilenet_v3_small.onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"])
                meta = sess.get_inputs()[0]
                self._session = sess
                self._in_name = meta.name
                self._onnx = True
                self._model = "onnx"
                return
            except ImportError:
                pass  # fallback на torch
        # приоритет 2: нативный onnxruntime-android AAR через pyjnius
        # (путь APK: AAR в gradle-зависимостях, pip-пакет onnxruntime не нужен)
        try:
            try:
                from jnius import autoclass
            except ImportError:
                from java import autoclass   # встроенный мост Chaquopy
            OrtEnvironment = autoclass("ai.onnxruntime.OrtEnvironment")
            self._ort_env = OrtEnvironment.getEnvironment()
            SessionOptions = autoclass("ai.onnxruntime.OrtSession$SessionOptions")
            self._ort_sess = self._ort_env.createSession(onnx_path, SessionOptions())
            self._jni = True
            self._model = "onnx-jni"
            return
        except Exception:
            self._jni = False
        try:
            import torch
            from torchvision.models import (mobilenet_v3_small,
                                            MobileNet_V3_Small_Weights)
        except ImportError:
            raise RuntimeError(
                "зрение недоступно: onnxruntime (py/jni), torch")
        w = MobileNet_V3_Small_Weights.DEFAULT
        # оффлайн-упаковка: веса рядом с проектом имеют приоритет
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "mobilenet_v3_small.pth")
        if os.path.exists(local):
            m = mobilenet_v3_small(weights=None)
            m.load_state_dict(torch.load(local, map_location="cpu"))
        else:
            m = mobilenet_v3_small(weights=w)  # скачается в кэш torch один раз
        m.classifier = torch.nn.Identity()   # 576-d pooled features
        m.eval()
        self._model = m
        self._tf = w.transforms()
        self._torch = torch

    # ---------- эмбеддинг ----------

    def embed_image(self, img) -> np.ndarray:
        """img: путь или PIL.Image -> нормированный 512-d вектор."""
        self._load()
        from PIL import Image
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if getattr(self, "_onnx", False):
            f = self._session.run(
                None, {self._in_name: self._preprocess(img)})[0][0]
        elif getattr(self, "_jni", False):
            f = self._run_jni(self._preprocess(img))
        else:
            x = self._tf(img).unsqueeze(0)
            with self._torch.no_grad():
                f = self._model(x).numpy()[0]
        v = self._proj.T @ f.astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed_image_tta(self, img) -> np.ndarray:
        """TTA: полный кадр + центральный кроп 70% (полевые снимки —
        нестабильное кадрирование). Усреднение нормированных эмбеддингов."""
        from PIL import Image
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        w, h = img.size
        crops = [img]
        if min(w, h) > 160:
            s = int(min(w, h) * 0.7)
            left, top = (w - s) // 2, (h - s) // 2
            crops.append(img.crop((left, top, left + s, top + s)))
        v = np.stack([self.embed_image(c) for c in crops]).mean(axis=0)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    def embed_batch(self, paths: list, bs: int = 8) -> np.ndarray:
        """Пачечный эмбеддинг (индексация): bs кадров за проход."""
        self._load()
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            chunk = paths[i:i + bs]
            if getattr(self, "_onnx", False):
                f = self._session.run(
                    None, {self._in_name: np.concatenate(
                        [self._preprocess(Image.open(p).convert("RGB"))
                         for p in chunk])})[0]
            elif getattr(self, "_jni", False):
                f = np.concatenate(
                    [self._run_jni(self._preprocess(Image.open(p).convert("RGB")))
                     for p in chunk])
            else:
                batch = [self._tf(Image.open(p).convert("RGB"))
                         for p in chunk]
                x = self._torch.stack(batch)
                with self._torch.no_grad():
                    f = self._model(x).numpy()
            for row in f:
                v = self._proj.T @ row.astype(np.float32)
                n = float(np.linalg.norm(v))
                out.append(v / n if n > 0 else v)
        return np.stack(out).astype(np.float32)

    def state_dict(self):
        return {"seed": self.seed}

    def load_state_dict(self, st):
        self.seed = int(st.get("seed", 13))
        self._proj = self._make_projection(self.seed)


if __name__ == "__main__":
    from PIL import Image
    e = ImageEmbedder()
    im = Image.fromarray(
        (np.random.default_rng(0).integers(0, 255, (300, 300, 3), np.uint8)))
    v = e.embed_image(im)
    assert v.shape == (512,) and abs(float(np.linalg.norm(v)) - 1) < 1e-4
    print("image_embedder: self-check OK (%.2f с/кадр на шуме — включая загрузку модели)"
          )
    import time
    t0 = time.time()
    for _ in range(3):
        e.embed_image(im)
    print("warm: %.2f с/кадр" % ((time.time() - t0) / 3))
