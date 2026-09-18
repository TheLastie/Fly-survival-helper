#!/data/data/com.termux/files/usr/bin/bash
# Установка приложения выживания на Android (Termux) — ШАГ 2.
# ШАГ 1: скопируйте папку проекта на телефон (USB / Syncthing / SD-карта).
# Запуск:  bash install_termux.sh
set -e
cd "$(dirname "$0")"

echo "== пакеты Termux =="
pkg update -y
pkg install -y python python-numpy cmake git clang
pip install pillow onnxruntime

echo "== модели =="
mkdir -p models
if [ ! -s models/mobilenet_v3_small.onnx ]; then
    echo "положите models/mobilenet_v3_small.onnx (ONNX-экспорт MobileNetV3)"
fi
# LLM ~1.1 ГБ; при нехватке места замените на qwen2.5-0.5b-instruct-q4_k_m.gguf
if [ ! -s qwen15b-q4.gguf ]; then
    curl -L -o qwen15b-q4.gguf \
      "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/qwen2.5-1.5b-instruct-q4_k_m.gguf"
fi

echo "== llama-server =="
if [ ! -x llama-server ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    cmake -S llama.cpp -B llama.cpp/build \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
    cmake --build llama.cpp/build --target llama-server -j"$(nproc)"
    cp llama.cpp/build/bin/llama-server .
    # разделяемые библиотеки рядом с бинарником
    cp llama.cpp/build/bin/*.so . 2>/dev/null || true
fi

cat > run.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# ЛАУНЧЕР: ./run.sh  — LLM + интерфейс всё в одном.
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=.
pkill llama-server 2>/dev/null || true
sleep 1
./llama-server -m qwen15b-q4.gguf --port 8081 -c 2048 -t 6 &
sleep 8
# CLI-диалог. Для браузера с загрузкой фото вместо последней строки:
#   python flybrain/webui.py --llama http://127.0.0.1:8081 --port 8321
python flybrain/cli.py --llama http://127.0.0.1:8081
EOF
chmod +x run.sh

echo
echo "Готово. Запуск: ./run.sh"
echo "Первая индексация баз (внутри диалога):"
echo "  индексируй виды plants_db"
echo "  индексируй виды fungi_db"
echo "  индексируй виды poisonous_db"
echo "  индексируй виды nature_db"
echo "  сохрани field_state"
echo "Далее: ./run.sh --load не нужен — состояние подхватится из field_state."
