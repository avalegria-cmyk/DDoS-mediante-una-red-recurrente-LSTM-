#!/usr/bin/env bash
set -euo pipefail

CSV_ORIGEN="${1:-}"
FLOW_DIR="${FLOW_DIR:-/opt/ddos-shield/data/flows}"

if [[ -z "$CSV_ORIGEN" || ! -f "$CSV_ORIGEN" ]]; then
  echo "Uso: $0 /ruta/archivo_cicflowmeter.csv"
  exit 1
fi

mkdir -p "$FLOW_DIR"
cp "$CSV_ORIGEN" "$FLOW_DIR/flow-$(date +%s).csv"
echo "CSV copiado a $FLOW_DIR. El detector lo procesara automaticamente."
