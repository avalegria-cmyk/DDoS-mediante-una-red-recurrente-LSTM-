#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Ejecuta este script con sudo para permitir captura real en macOS."
  exit 1
fi

DETECTOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$DETECTOR_DIR/.venv/bin/python}"
PORT="${PORT:-4100}"
VICTIM_PORT="${VICTIM_PORT:-8080}"
VICTIM_DIR="${VICTIM_DIR:-$DETECTOR_DIR/victima}"
VICTIM_ENABLED="${VICTIM_ENABLED:-true}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "No existe $PYTHON_BIN. Ejecuta primero:"
  echo "  cd '$DETECTOR_DIR' && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements_rf11.txt"
  exit 1
fi

mkdir -p \
  "$DETECTOR_DIR/data/flows" \
  "$DETECTOR_DIR/data/pcap" \
  "$DETECTOR_DIR/data/state" \
  "$DETECTOR_DIR/logs"

EXISTING_PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$EXISTING_PIDS" ]]; then
  kill $EXISTING_PIDS
  sleep 1
fi

if [[ "$VICTIM_ENABLED" == "true" ]]; then
  EXISTING_VICTIM_PIDS="$(lsof -tiTCP:"$VICTIM_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$EXISTING_VICTIM_PIDS" ]]; then
    kill $EXISTING_VICTIM_PIDS
    sleep 1
  fi

  nohup "$PYTHON_BIN" -m http.server "$VICTIM_PORT" --bind 0.0.0.0 --directory "$VICTIM_DIR" \
    > "$DETECTOR_DIR/logs/victima_8080.log" 2>&1 &
  echo $! > "$DETECTOR_DIR/logs/victima_8080.pid"
fi

export MODEL_KIND="xgboost"
export MODEL_DIR="$DETECTOR_DIR/modelo_rf_multiclase"
export FLOW_DIR="$DETECTOR_DIR/data/flows"
export STATE_DIR="$DETECTOR_DIR/data/state"
export PCAP_DIR="$DETECTOR_DIR/data/pcap"
export WEB_DIR="$DETECTOR_DIR/web"
export CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-en0}"
export CAPTURE_SECONDS="${CAPTURE_SECONDS:-15}"
export CAPTURE_AUTO_START="${CAPTURE_AUTO_START:-true}"
export CAPTURE_USE_SUDO="false"
export AUTO_BLOCK="${AUTO_BLOCK:-false}"
export PROCESS_EXISTING_FILES="${PROCESS_EXISTING_FILES:-false}"
export KEEP_CAPTURE_FILES="${KEEP_CAPTURE_FILES:-false}"
export PREDICTION_THRESHOLD="${PREDICTION_THRESHOLD:-0.65}"
export RATE_ATTACK_PPS="${RATE_ATTACK_PPS:-1000}"
export RATE_ATTACK_ANY_PORT="${RATE_ATTACK_ANY_PORT:-true}"
export MODEL_ATTACK_MIN_PPS="${MODEL_ATTACK_MIN_PPS:-10}"
export CICFLOWMETER_CMD="'$PYTHON_BIN' '$DETECTOR_DIR/scripts/convertir_pcap.py' {pcap} {csv}"

nohup "$PYTHON_BIN" -m uvicorn app:app --app-dir "$DETECTOR_DIR" --host 0.0.0.0 --port "$PORT" \
  > "$DETECTOR_DIR/logs/rf11_detector_admin_mac.log" 2>&1 &

echo $! > "$DETECTOR_DIR/logs/rf11_detector_admin_mac.pid"
echo "Detector RF 11 caracteristicas iniciado: http://127.0.0.1:$PORT"
if [[ "$VICTIM_ENABLED" == "true" ]]; then
  echo "Victima HTTP iniciada: http://127.0.0.1:$VICTIM_PORT"
fi
echo "Modelo: $MODEL_DIR/xgboost.pkl"
echo "Clases: Benign, DNS, LDAP, MSSQL, NTP, NetBIOS, Portmap, SNMP, Syn, TFTP, UDP, UDPLag, WebDDoS"
echo "Captura: $CAPTURE_INTERFACE cada ${CAPTURE_SECONDS}s"
echo "Regla por tasa: >= ${RATE_ATTACK_PPS} pps en cualquier puerto"
echo "Filtro anti-falsos positivos: modelo ignorado bajo ${MODEL_ATTACK_MIN_PPS} pps"
echo "Archivos de captura: ${KEEP_CAPTURE_FILES}"
