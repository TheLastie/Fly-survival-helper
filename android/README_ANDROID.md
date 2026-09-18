# FlyBrain Survival — APK из ОДНОГО файла

Цель: пользователь переносит на телефон ровно один .apk, устанавливает,
открывает — всё работает. Без Termux, без закачек, без индексаций.

## Что уже внутри APK (assets)
- `state/` — готовое полевое состояние: 4027 документов, 3655 фото,
  117 видов (растения/грибы/ядовитые/природа), ~11 МБ;
- `models/mobilenet_v3_small.onnx` — зрение, 3.7 МБ;
- весь код проекта (30 модулей) в `python/flybrain/`.

Первый запуск: распаковка state из assets (~2 с) → Web-UI на
127.0.0.1:8321 → WebView. Никакой сети не требуется.

## Сборка (Android Studio)
1. SDK 34, NDK; открыть папку `android/`.
2. Sync gradle (Chaquopy подтянет Python 3.11 + numpy/pillow/onnxruntime).
3. Run на телефоне (отладка USB). Готово.

## Размер base APK (прогноз)
| Слой | МБ |
|---|---|
| Chaquopy + Python 3.11 | ~35 |
| numpy + pillow + onnxruntime | ~40–55 |
| WebView-оболочка (Kotlin) | ~5 |
| Модель ONNX | 4 |
| Код | 1 |
| **State (базы)** | **11** |
| **итого** | **~95–110 МБ** |

## LLM внутри APK (опционально, v2)
1. Соберите llama.cpp под arm64 (нужен NDK):
   ```
   git clone --depth 1 https://github.com/ggerganov/llama.cpp
   cmake -S llama.cpp -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
   cmake --build build-android --target llama -j
   # получится build-android/bin/libllama.so
   ```
2. Положите `libllama.so` в `app/src/main/python/` и
   `qwen15b-q4.gguf` (1.1 ГБ) в `app/src/main/assets/` (или докачка
   первым запуском из самого приложения — тогда APK без неё).
3. `llama_local.py` подцепит их автоматически; без них — безопасные
   шаблоны (предупреждения о ядовитости работают всегда, они в коде).

С LLM APK ~1.2 ГБ (sideload — без проблем; в Play — через asset packs).

## Проверено / не проверено
- Проверено здесь: синтаксис Python-файлов, структура, консистентность
  с остальным проектом (66/66 тестов ядра зелёные).
- НЕ проверено (нет Android SDK в среде): первая gradle-сборка, работа
  Chaquopy-pip с onnxruntime (запасной путь — нативный AAR + pyjnius,
  точка замены изолирована в image_embedder._load), ctypes-вызовы к
  libllama.so (подписи сверены с llama.h, но первый прогон на устройстве
  может потребовать правку CtxParams под версию llama.cpp).

## Получить APK файл — два пути

### Путь А (без установки чего-либо): GitHub Actions
1. Создайте репозиторий на github.com и залейте туда всю папку проекта
   (кнопка upload / git push).
2. Откройте вкладку **Actions** → workflow `build-apk` → **Run workflow**.
3. Через ~10–15 минут во вкладке **Actions → build-apk → Artifacts**
   появится `flybrain-survival-debug.zip` — внутри готовый .apk.
   Скачайте на телефон → установить → готово.

### Путь Б: Android Studio
Открыть папку `android/` как проект → Sync gradle → Build → Build APK(s)
→ app-debug.apk лежит в android/app/build/outputs/apk/debug/.

### Где APK НЕ лежит
В архиве flybrain_project.zip APK нет — там исходники проекта APK.
