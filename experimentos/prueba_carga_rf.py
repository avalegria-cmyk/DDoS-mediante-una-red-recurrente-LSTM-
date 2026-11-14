"""Prueba de carga del Random Forest compacto DESPLEGADO (correccion C).

Sustituye la prueba de carga que el articulo hacia sobre el LSTM. Mide, para lotes
de tamano creciente, la latencia por flujo, el throughput, el uso de CPU y la RAM
del mismo camino que usa app.py: imputacion + escalado + predict_proba.

Con --pcap tambien mide la etapa de extraccion de flujos (CICFlowMeter 0.5.0) para
reportar el tiempo de la cadena captura -> flujos -> inferencia. La etapa de la
regla de firewall se mide en la prueba de mitigacion con el sistema levantado.

Uso
---
    python3 experimentos/prueba_carga_rf.py --archive-dir RUTA/archive
    python3 experimentos/prueba_carga_rf.py --archive-dir RUTA/archive --pcap captura.pcap
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "modelo_rf_multiclase"
DEFAULT_OUTPUT_DIR = ROOT / "experimentos" / "resultados"
DEFAULT_BATCHES = [1, 10, 100, 1000, 5000, 10000, 50000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--archive-dir", type=Path, default=ROOT.parent / "archive",
                   help="Carpeta con los .parquet; se usan flujos reales de los archivos *-testing")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    p.add_argument("--repeats", type=int, default=20, help="Repeticiones por tamano de lote")
    p.add_argument("--pcap", type=Path, default=None, help="PCAP para medir tambien la extraccion de flujos")
    return p.parse_args()


def load_model(model_dir: Path):
    pkl, gz = model_dir / "xgboost.pkl", model_dir / "xgboost.pkl.gz"
    if pkl.exists():
        model = joblib.load(pkl)
    else:
        with gzip.open(gz, "rb") as fh:
            model = joblib.load(fh)
    preprocessors = joblib.load(model_dir / "preprocessors.pkl")
    feature_cols = json.loads((model_dir / "feature_cols.json").read_text(encoding="utf-8"))
    return model, preprocessors, feature_cols


def load_flows(archive_dir: Path, feature_cols: list[str], needed: int) -> tuple[pd.DataFrame, str]:
    files = sorted(archive_dir.glob("*-testing.parquet")) or sorted(archive_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No hay archivos .parquet en {archive_dir}. La prueba de carga debe usar flujos reales del dataset.")
    per_file = max(needed // len(files) + 1, 1)
    parts = []
    for path in files:
        df = pd.read_parquet(path, columns=feature_cols)
        parts.append(df.sample(n=min(per_file, len(df)), random_state=42))
    flows = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(flows) < needed:  # se repiten flujos reales si el lote pedido es mayor que los disponibles
        flows = pd.concat([flows] * (needed // len(flows) + 1), ignore_index=True)
    return flows.iloc[:needed].reset_index(drop=True), f"{len(files)} archivos de {archive_dir}"


def infer(model, preprocessors, feature_cols: list[str], df: pd.DataFrame) -> np.ndarray:
    """Mismo camino que prepare_xgboost_features + predict_proba en app.py."""
    values = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64", copy=True)
    values[~np.isfinite(values)] = np.nan  # inf -> NaN, igual que en app.py
    X = pd.DataFrame(values, columns=feature_cols)
    X_imp = pd.DataFrame(preprocessors["imputer"].transform(X), columns=feature_cols)
    X_scaled = pd.DataFrame(preprocessors["scaler"].transform(X_imp), columns=feature_cols)
    return model.predict_proba(X_scaled)


def measure_batches(model, preprocessors, feature_cols, flows, batches, repeats) -> pd.DataFrame:
    proc = psutil.Process()
    cores = psutil.cpu_count(logical=True) or 1
    infer(model, preprocessors, feature_cols, flows.iloc[:100])  # calentamiento
    rows = []
    for size in batches:
        batch = flows.iloc[:size]
        times = []
        proc.cpu_percent(None)
        wall0 = time.perf_counter()
        for _ in range(repeats):
            t0 = time.perf_counter()
            infer(model, preprocessors, feature_cols, batch)
            times.append(time.perf_counter() - t0)
        wall = time.perf_counter() - wall0
        cpu = proc.cpu_percent(None)  # % de un nucleo, acumulado en todos los hilos
        med = statistics.median(times)
        rows.append({
            "flujos_por_lote": size,
            "latencia_lote_ms_mediana": 1000 * med,
            "latencia_lote_ms_p95": 1000 * float(np.percentile(times, 95)),
            "latencia_por_flujo_ms": 1000 * med / size,
            "throughput_flujos_s": size / med,
            "cpu_pct_de_un_nucleo": cpu,
            "cpu_pct_del_equipo": cpu / cores,
            "ram_mb": proc.memory_info().rss / 1e6,
            "repeticiones": repeats,
            "duracion_total_s": wall,
        })
        r = rows[-1]
        print(f"  lote {size:6d}: {r['latencia_por_flujo_ms']:.5f} ms/flujo | {r['throughput_flujos_s']:10.0f} flujos/s | "
              f"CPU {r['cpu_pct_del_equipo']:.1f} % | RAM {r['ram_mb']:.0f} MB")
    return pd.DataFrame(rows)


def measure_pcap(pcap: Path, model, preprocessors, feature_cols, out_dir: Path) -> dict:
    """Tiempo de extraccion de flujos (cicflowmeter) + normalizacion de columnas + inferencia."""
    csv_path = out_dir / f"{pcap.stem}_flows.csv"
    csv_path.unlink(missing_ok=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "convertir_pcap.py"), str(pcap), str(csv_path)], check=True)
    extract_s = time.perf_counter() - t0

    sys.path.insert(0, str(ROOT))
    import app  # reutiliza el mapeo de columnas del sistema desplegado

    t0 = time.perf_counter()
    df = app.normalize_cicflowmeter_columns(pd.read_csv(csv_path))
    infer(model, preprocessors, feature_cols, df)
    infer_s = time.perf_counter() - t0
    n = len(df)
    return {
        "pcap": str(pcap), "pcap_mb": pcap.stat().st_size / 1e6, "flujos": int(n),
        "extraccion_flujos_s": extract_s, "lectura_normalizacion_inferencia_s": infer_s,
        "total_s": extract_s + infer_s,
        "ms_por_flujo_extraccion": 1000 * extract_s / max(n, 1),
        "ms_por_flujo_inferencia": 1000 * infer_s / max(n, 1),
    }


def plot(table: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = table["flujos_por_lote"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, table["latencia_por_flujo_ms"], "o-", color="tab:blue")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Flows per batch"); ax.set_ylabel("Latency per flow (ms)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(x, table["throughput_flujos_s"], "s--", color="tab:red")
    ax2.set_yscale("log"); ax2.set_ylabel("Throughput (flows/s)", color="tab:red")
    ax.set_title("Compact Random Forest: latency and throughput")
    fig.tight_layout(); fig.savefig(out_dir / "figura_rf_latencia_throughput.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, table["cpu_pct_del_equipo"], "o-", color="tab:green")
    ax.set_xscale("log"); ax.set_xlabel("Flows per batch"); ax.set_ylabel("CPU usage (% of machine)", color="tab:green")
    ax2 = ax.twinx()
    ax2.plot(x, table["ram_mb"], "s--", color="tab:purple")
    ax2.set_ylabel("RAM (MB)", color="tab:purple")
    ax.set_title("Compact Random Forest: CPU and RAM")
    fig.tight_layout(); fig.savefig(out_dir / "figura_rf_cpu_ram.png", dpi=200); plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocessors, feature_cols = load_model(args.model_dir.expanduser().resolve())
    print(f"Modelo: {type(model).__name__} | arboles: {getattr(model, 'n_estimators', '?')} | "
          f"variables: {len(feature_cols)} | clases: {len(getattr(model, 'classes_', []))}")
    flows, origin = load_flows(args.archive_dir.expanduser().resolve(), feature_cols, max(args.batches))
    print(f"Flujos de prueba: {len(flows)} ({origin})\n--- PRUEBA DE CARGA ---")

    table = measure_batches(model, preprocessors, feature_cols, flows, args.batches, args.repeats)
    table.to_csv(out_dir / "carga_rf.csv", index=False)
    plot(table, out_dir)

    payload = {
        "modelo": type(model).__name__, "n_estimators": getattr(model, "n_estimators", None),
        "variables": feature_cols, "origen_flujos": origin,
        "entorno": {"maquina": platform.platform(), "procesador": platform.processor(),
                    "nucleos_logicos": psutil.cpu_count(logical=True),
                    "ram_total_gb": psutil.virtual_memory().total / 1e9, "python": platform.python_version()},
        "lotes": table.to_dict(orient="records"),
    }
    if args.pcap:
        print("--- CADENA PCAP -> FLUJOS -> INFERENCIA ---")
        payload["pcap"] = measure_pcap(args.pcap.expanduser().resolve(), model, preprocessors, feature_cols, out_dir)
        print(json.dumps(payload["pcap"], indent=2, ensure_ascii=False))
    (out_dir / "carga_rf.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resultados en {out_dir}: carga_rf.csv, carga_rf.json, figura_rf_latencia_throughput.png, figura_rf_cpu_ram.png")


if __name__ == "__main__":
    main()
