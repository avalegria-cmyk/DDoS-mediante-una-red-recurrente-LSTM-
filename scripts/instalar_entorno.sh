#!/usr/bin/env bash
set -euo pipefail

DETECTOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DETECTOR_DIR"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements_rf11.txt

if [[ ! -f modelo_rf_multiclase/xgboost.pkl && -f modelo_rf_multiclase/xgboost.pkl.gz ]]; then
  gzip -dk modelo_rf_multiclase/xgboost.pkl.gz
fi

echo "Entorno listo en: $DETECTOR_DIR/.venv"
