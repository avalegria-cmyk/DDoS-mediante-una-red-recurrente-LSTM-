from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ["MODEL_KIND"] = "lstm"
os.environ["AUTO_BLOCK"] = "false"
os.environ["MODEL_DIR"] = str(ROOT / "modelo")
os.environ["FLOW_DIR"] = str(ROOT / "data" / "flows")
os.environ["STATE_DIR"] = str(ROOT / "data" / "state")
os.environ["PCAP_DIR"] = str(ROOT / "data" / "pcap")
sys.path.insert(0, str(ROOT))

import app as detector


def main() -> None:
    if len(detector.feature_cols) != 47:
        raise RuntimeError(f"Se cargaron {len(detector.feature_cols)} variables; se esperaban 47")
    if detector.timesteps != 10:
        raise RuntimeError(f"Se cargaron {detector.timesteps} timesteps; se esperaban 10")
    if detector.scaler_lstm.mean_.shape[0] != 47:
        raise RuntimeError("El scaler no coincide con las 47 variables")

    raw_features = np.tile(detector.scaler_lstm.mean_, (detector.timesteps, 1))
    flows = pd.DataFrame(raw_features, columns=detector.feature_cols)
    flows["Src IP"] = "198.51.100.25"
    flows["Dst IP"] = "192.0.2.10"
    flows["Dst Port"] = 8080
    flows["Timestamp"] = "2026-07-23 12:00:00"

    detector.process_sequence_file(Path("pipeline-lstm-reales.csv"), flows)
    events = list(detector.state.traffic)
    if len(events) != detector.timesteps:
        raise RuntimeError(f"Se esperaban {detector.timesteps} eventos y se obtuvieron {len(events)}")

    latest = events[0]
    if latest["modelKind"] != "lstm" or latest["windowSize"] != 10:
        raise RuntimeError("La ventana temporal no llego correctamente al modelo")
    if not np.isfinite(latest["probability"]):
        raise RuntimeError("La probabilidad del LSTM no es finita")

    snake_row = {source: 1 for source in detector.PY_CICFLOWMETER_COLUMNS}
    normalized = detector.normalize_cicflowmeter_columns(pd.DataFrame([snake_row]))
    missing = [column for column in detector.feature_cols if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"El adaptador CICFlowMeter no produce estas columnas reales: {missing}")

    print(f"OK LSTM real: 10 flujos, 47 variables, probabilidad {latest['probability']}")


if __name__ == "__main__":
    main()
