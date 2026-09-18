# Playbook: Python-приложения в APK без боли

Собрано из 13 итераций отладки реального проекта (Chaquopy + WebView + numpy +
ONNX на Nothing Phone 2a, Android 16). Каждое правило здесь — из реального
краша, который стоил часов. Проверяйте пункты по порядку: 90% «тихих крашей»
лежат в первых трёх разделах.

---

## 0. Железные правила процесса (они важнее кода)

1. **Пуш обязан быть подтверждён.** После `git push` всегда проверяйте
   `git ls-remote origin main` — если хэш не совпал с локальным HEAD,
   сборка в CI пойдёт по СТАРОМУ коду, а вы будете отлаживать призрака.
   Несвязанные истории (re-init репозитория) → push молча отклоняется.
   Лечение: `git fetch && git reset --hard origin/main` перед работой,
   `--force` только осознанно.
2. **Один файл = одна версия.** Никогда не перезаписывайте APK одним именем
   (`app-debug.apk` → `App-v1.apk`, `App-v2.apk`...). Иначе file manager /
   мессенджер отдаст старый файл из кэша, и вы протестируете его N раз.
3. **Хэшируйте каждую сборку**: `sha256sum App-vN.apk` — ведите таблицу.
   Подозрение «присылают один и тот же файл» проверяется за 10 секунд.
4. **Окружение может терять файлы между шагами.** После любого долгого
   действия проверяйте `ls` критичных путей. Работайте в `/tmp`, коммитьте
   рано, пушьте рано.
5. **CI — единственный честный источник APK.** Локальная «сборка» без
   Android Studio недостижима; GitHub Actions бесплатен и воспроизводим.

## 1. Главный баг всей истории: Kotlin-плагин

Симптом: приложение падает МГНОВЕННО при запуске, ни одного экрана, ни
одного перехвата, на всех версиях.

Причина: в `build.gradle` не было `org.jetbrains.kotlin.android`. AGP
**молча игнорирует все `.kt`-файлы** — класса MainActivity в APK не
существует, система падает при создании стартовой активности. Сборка при
этом УСПЕШНА (aapt не проверяет существование классов).

**Обязательный минимум в скелете:**

```gradle
// build.gradle (root)
plugins {
    id 'com.android.application' version '8.5.0' apply false
    id 'org.jetbrains.kotlin.android' version '1.9.24' apply false   // ← без него тишина и краш
}

// app/build.gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}
android {
    compileOptions {                    // ← иначе: Inconsistent JVM-target
        sourceCompatibility JavaVersion.VERSION_17                   //   Java 1.8 vs Kotlin 17
        targetCompatibility JavaVersion.VERSION_17
    }
}
```

**Проверка, что класс реально в APK** (делать ПОСЛЕ каждой сборки):

```bash
python3 - <<'EOF'
import zipfile
zf = zipfile.ZipFile('App.apk')
for n in zf.namelist():
    if n.endswith('.dex'):
        d = zf.read(n)
        if b'MainActivity' in d:
            print('OK: MainActivity в', n); break
else:
    print('КЛАССА НЕТ — Kotlin не компилировался!')
EOF
```

## 2. Нативные библиотеки и версии ОС

- **Android 15/16 → 16KB-страницы памяти.** Нативные `.so`, собранные под
  8KB, не загружаются: процесс умирает за миллисекунды на `dlopen`, до
  первого кадра. Берите свежие версии всего нативного: NDK 27+,
  onnxruntime-android ≥1.20, Chaquopy ≥16.
- `android:extractNativeLibs="true"` в манифест — без него Chaquopy
  не грузит libpython (классический мгновенный краш).
- `ndk { abiFilters "arm64-v8a" }` — режем лишние ABI, экономим 30 МБ.
- minSdk 24, targetSdk 34 работают на Android 16 нормально.

## 3. Chaquopy: Python внутри APK

- Плагин `com.chaquo.python` version 16.0.0, Gradle пиновать 8.10
  (Chaquopy несовместим с Gradle 9: `VersionNumber` удалён).
- **Не тяните pip-пакеты без колёс.** `pyjnius` с PyPI — sdist с C-кодом,
  Chaquopy запрещает компиляцию. У Chaquopy есть встроенный Java-мост:
  `from java import autoclass` — его достаточно.
- **НЕ работайте с Android API из Python при старте.** AssetManager,
  распаковка файлов, копирование — делайте в Kotlin (Java-код в APK),
  Python получайте готовые файлы через `filesDir`. Иначе цепочка
  `import android` / `com.chaquo.python` / версионные различия мостов
  даёт по одному `ModuleNotFoundError` за сборку.
- Манифест после слияния проверяйте: чужой `ContentProvider` от AAR-
  зависимости (androidx.startup) может грузить `.so` на старте процесса.

## 4. Диагностика без компьютера (главное достижение проекта)

Пользователь в тайге: ни adb, ни логов. Архитектура самодиагностики:

1. **Аварийный сервер**: `try: start() except:` → пишем traceback в файл
   и поднимаем микро-HTTP-сервер на 127.0.0.1:PORT, отдающий traceback
   HTML-страницей. WebView показывает его вместо закрытия приложения.
2. **Кнопка «Отправить лог»** на том экране: `Intent.ACTION_SEND` с текстом
   → любой мессенджер. Один тап — лог у разработчика.
3. **Пошаговый статус запуска**: каждый этап Python пишет в TextView
   (переданный Java-объект вызывается прямо из Python: `status.setText`).
   Нативный краш ЗАМОРАЖИВАЕТ последний текст — локализация без logcat.
4. Retry-цикл WebView: сервер стартует дольше WebView (распаковка
   ассетов), перезагружайте страницу до N раз с интервалом 1 с.
5. Бисекция: сборка без Python и без AAR (экран «B1 OK») отвечает на
   вопрос «код или конфигурация» за одну итерацию.

Шаблон экрана ошибки есть в репозитории: `MainActivity.kt` (showFatal),
`android_host.py` (crash-server).

## 5. CI: GitHub Actions workflow

```yaml
# .github/workflows/build-apk.yml
- uses: actions/setup-java@v4
  with: { distribution: temurin, java-version: 17 }
# НЕ использовать android-actions/setup-android — сломан на новых runner'ах
# (образ ubuntu-latest уже содержит SDK):
- run: yes | $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --licenses
- uses: gradle/actions/setup-gradle@v4        # пин 8.10
  with: { gradle-version: "8.10" }
- run: cd android && gradle assembleDebug --stacktrace
- uses: actions/upload-artifact@v4
  with: { name: app-v${{ github.run_number }}, path: android/app/build/outputs/apk/debug/*.apk }
```

Подводные камни, пройденные за вас: setup-android против cmdline-tools 16;
`implementation(...)` внутри блока `android {}` (должен быть в `dependencies`);
fine-grained PAT не создаёт репозитории и пишет 404 вместо 403 (права
проверять в web UI); `sourceSets {}` внутри `defaultConfig` невалиден.

## 6. Подпись и установка

- Каждая CI-сборка = новый debug-ключ → `INSTALL_FAILED_UPDATE_INCOMPATIBLE`
  при установке поверх. Решение: фиксированный keystore в репозитории
  (`keytool -genkeypair ...`, signingConfigs.debug на него). Debug-ключ
  в открытом виде — приемлемо.
- APK до ~100 МБ — норма для sideload; Play требует AAB/asset packs.

## 7. Размер APK и почему он «не меняется»

Вес = константы (Chaquopy-рантайм ~30 МБ, numpy ~15 МБ, AAR onnxruntime
~10 МБ, состояние ~11 МБ, модель 4 МБ) + ваш код (~0.3 МБ). Правки кода
невидимы на фоне мегабайт — поэтому «размер тот же» ≠ «файл тот же».
Проверка только хэшем. Ужимка: minifyEnabled+shrinkResources для release,
strip-символы, отказ от AAR.

## 8. Чек-лист нового проекта (копировать и проверять)

- [ ] Kotlin-плагин в обоих build.gradle + compileOptions JVM 17
- [ ] MainActivity проверен в dex сразу после первой сборки
- [ ] Пуш подтверждён `git ls-remote` после КАЖДОГО push
- [ ] Хэш APK записан; имя файла содержит версию
- [ ] extractNativeLibs, minSdk 24, arm64-v8a
- [ ] Нативные зависимости свежие (16KB!)
- [ ] Распаковка assets — в Kotlin, не в Python
- [ ] Аварийный экран + share-кнопка + пошаговый статус
- [ ] Workflow: без setup-android, Gradle 8.10, upload-artifact
- [ ] Фиксированный keystore; APK носит уникальное имя

---

*Проект: FlyBrain Survival. 13 CI-итераций от «тихого краша» до рабочего
приложения. Главный урок: когда «всё падает без причин» — проверяй сначала
то, что молчит: плагины, пуши, хэши.*
