#!/usr/bin/env bash
# Crea o quita direcciones de prueba en la interfaz loopback para que atacante, victima y
# clientes legitimos tengan IP distintas dentro de la misma maquina (ninguna en lista blanca).
#   sudo ./experimentos/red_prueba.sh up
#   sudo ./experimentos/red_prueba.sh down
set -euo pipefail

ACTION="${1:-}"
IPS=(10.99.0.1 10.99.0.2 10.99.0.11 10.99.0.12 10.99.0.13 10.99.0.14 10.99.0.15)

case "$ACTION" in
  up)
    for ip in "${IPS[@]}"; do
      /sbin/ifconfig lo0 alias "$ip" netmask 255.255.255.255
    done
    echo "Direcciones de prueba creadas en lo0: ${IPS[*]}"
    ;;
  down)
    for ip in "${IPS[@]}"; do
      /sbin/ifconfig lo0 -alias "$ip" 2>/dev/null || true
    done
    echo "Direcciones de prueba eliminadas de lo0."
    ;;
  *)
    echo "Uso: sudo $0 up|down" >&2
    exit 1
    ;;
esac
