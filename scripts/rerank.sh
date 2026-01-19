#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME=/opt/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"

# llamacpp 빌드 경로
BUILD_BIN="/data/ephemeral/home/build/llama.cpp/bin"
export BUILD_BIN

# libmtmd/libggml/libllama 위치가 BUILD_BIN 안이라서 추가
export LD_LIBRARY_PATH="$BUILD_BIN:$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

MODEL="/huggingface/models--gpustack--bge-reranker-v2-m3-GGUF/snapshots/3093af03b1a635e67b084b1d8c03c5f5e020fd05/bge-reranker-v2-m3-FP16.gguf"
export MODEL

# ngl ALl: 25/25
NGL="${NGL:-25}"
CTX="${CTX:-3072}"
PORT="${PORT:-30410}"
export NGL CTX PORT

# 원하는 로그파일: 기본 server-rerank.log
LOG="${LOG:-log/server-rerank.log}"
PIDFILE="${PIDFILE:-log/llama-server-rerank.pid}"

mkdir -p "$(dirname "$LOG" 2>/dev/null || echo .)"


# 파이프라인 전체를 백그라운드로(세션 종료에도 유지)
start() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running: pid=$(cat "$PIDFILE")"
    echo "Log: $LOG"
    exit 0
  fi

  : > "$LOG"
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8

  nohup "$BUILD_BIN/llama-server" \
    -m "$MODEL" \
    -c "$CTX" \
    -b "$CTX" \
    -ub "$CTX" \
    -ngl "$NGL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --parallel 1 \
    --embedding \
    --pooling rank \
    --rerank \
    --cache-ram 0 \
    --timeout 3600 \
    </dev/null \
    > >(TZ=Asia/Seoul ts "%Y-%m-%d %H:%M:%S" >>"$LOG") 2>&1 &

  echo $! >"$PIDFILE"
  echo "Started: pid=$(cat "$PIDFILE")"
  echo "Log: $LOG"
}

stop() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")"
    echo "Stopped: pid=$(cat "$PIDFILE")"
    rm -f "$PIDFILE"
  else
    echo "Not running (no valid pid)."
  fi
}

status() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Running: pid=$(cat "$PIDFILE")"
    echo "Log: $LOG"
  else
    echo "Not running."
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "Usage: $0 {start|stop|restart|status}" ; exit 2 ;;
esac
