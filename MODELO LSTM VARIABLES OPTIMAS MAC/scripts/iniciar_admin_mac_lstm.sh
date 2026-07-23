#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Ejecuta este script con sudo para permitir captura real en macOS."
  exit 1
fi

DETECTOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$(cd "$DETECTOR_DIR/.." && pwd)"
if [[ -x "$DETECTOR_DIR/.venv/bin/python" ]]; then
  VENV="$DETECTOR_DIR/.venv"
else
  VENV="$PROJECT_DIR/implementacion/.venv"
fi
RUN_DIR="$PROJECT_DIR/.run"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$DETECTOR_DIR/data/flows" "$DETECTOR_DIR/data/pcap" "$DETECTOR_DIR/data/state"

EXISTING_PIDS="$(lsof -tiTCP:4100 -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$EXISTING_PIDS" ]]; then
  kill $EXISTING_PIDS
  sleep 1
fi

export MODEL_KIND="lstm"
export MODEL_DIR="$DETECTOR_DIR/modelo"
export FLOW_DIR="$DETECTOR_DIR/data/flows"
export STATE_DIR="$DETECTOR_DIR/data/state"
export PCAP_DIR="$DETECTOR_DIR/data/pcap"
export WEB_DIR="$DETECTOR_DIR/web"
export CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-en0}"
export CAPTURE_SECONDS="${CAPTURE_SECONDS:-15}"
export CAPTURE_AUTO_START="false"
export CAPTURE_USE_SUDO="false"
export AUTO_BLOCK="false"
export CICFLOWMETER_CMD="'$VENV/bin/python' '$DETECTOR_DIR/scripts/convertir_pcap.py' {pcap} {csv}"

nohup "$VENV/bin/python" -m uvicorn app:app --app-dir "$DETECTOR_DIR" --host 0.0.0.0 --port 4100 \
  > "$LOG_DIR/lstm_detector_admin_mac.log" 2>&1 &

echo $! > "$RUN_DIR/lstm_detector_admin_mac.pid"
echo "Detector LSTM con captura real iniciado: http://127.0.0.1:4100"
echo "Modelo: $MODEL_DIR/modelo_lstm_ddos.keras"
echo "Variables: 47"
