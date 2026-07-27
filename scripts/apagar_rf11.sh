#!/usr/bin/env bash
set -euo pipefail

DETECTOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-4100}"
VICTIM_PORT="${VICTIM_PORT:-8080}"

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 1
  fi
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
  fi
}

kill_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(sed -n '1p' "$pid_file")"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

kill_pid_file "$DETECTOR_DIR/logs/rf11_detector_admin_mac.pid"
kill_pid_file "$DETECTOR_DIR/logs/victima_8080.pid"
kill_port "$PORT"
kill_port "$VICTIM_PORT"

echo "Apagado solicitado para detector puerto $PORT y victima puerto $VICTIM_PORT."
