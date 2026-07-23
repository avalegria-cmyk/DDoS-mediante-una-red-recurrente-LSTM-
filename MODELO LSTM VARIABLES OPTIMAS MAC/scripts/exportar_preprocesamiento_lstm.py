from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
LSTM_DIR = ROOT / "LSTMDDOS"
INFO_PATH = LSTM_DIR / "modelos_lstm_gru" / "info_experimento_lstm_gru.json"
MODEL_DIR = ROOT / "implementacion" / "detector_real" / "modelo"


def main() -> None:
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    feature_cols = info["features"]
    max_rows_per_file = int(info.get("max_rows_per_file", 12000))

    frames = []
    for parquet_path in sorted((LSTM_DIR / "archive").glob("*-training.parquet")):
        df = pd.read_parquet(parquet_path)
        if len(df) > max_rows_per_file:
            df = df.sample(n=max_rows_per_file, random_state=42).sort_index()
        frames.append(df[feature_cols])
        print(f"{parquet_path.name}: {len(df)} filas")

    X_train = pd.concat(frames, ignore_index=True)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler_lstm_gru.pkl")
    (MODEL_DIR / "feature_cols_lstm_gru.json").write_text(
        json.dumps(feature_cols, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "model_kind": "lstm",
        "timesteps": int(info.get("timesteps", 10)),
        "input_shape": info.get("input_shape", [10, len(feature_cols)]),
        "feature_count": len(feature_cols),
        "scaler": "scaler_lstm_gru.pkl",
        "feature_cols": "feature_cols_lstm_gru.json",
        "source_info": str(INFO_PATH),
    }
    (MODEL_DIR / "lstm_runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Scaler LSTM/GRU exportado en:", MODEL_DIR)


if __name__ == "__main__":
    main()
