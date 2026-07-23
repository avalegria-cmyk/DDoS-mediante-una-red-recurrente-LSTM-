#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CAPTURE_INTERFACE="lo0"
export RATE_ATTACK_PPS="500"

echo "Modo prueba local activado: captura en lo0."
echo "Usa este modo cuando ataques desde este mismo Mac al servidor victima."
echo "Para ataques desde otra maquina de la red usa iniciar_admin_mac_lstm.sh."

exec "$SCRIPT_DIR/iniciar_admin_mac_lstm.sh"
