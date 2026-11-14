"""Medicion integral de la cadena captura -> flujos -> inferencia -> regla, por etapas.

Para cada captura real guardada de las pruebas de mitigacion mide:
  * ventana de captura: duracion real y numero de paquetes;
  * extraccion de flujos (CICFlowMeter 0.5.0): tiempo, tiempo de CPU y memoria maxima;
  * inferencia y decision (lectura del CSV, normalizacion, imputacion, escalado, Random Forest,
    regla por tasa y decision por flujo, con el mismo codigo de app.py): tiempo, CPU y memoria;
  * insercion de la regla en pf: tiempos reales registrados en la bitacora del detector.

Uso: .venv/bin/python experimentos/medir_cadena.py NOMBRE=RUTA.pcap [NOMBRE=RUTA.pcap ...]
"""
from __future__ import annotations

import json
import os
import re
import resource
import statistics as st
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="cadena_"))
os.environ.setdefault("STATE_DIR", str(TMP / "state"))  # la bitacora real no se toca
os.environ.setdefault("FLOW_DIR", str(TMP / "flows"))
os.environ.setdefault("AUTO_BLOCK", "false")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scapy.utils import PcapReader  # noqa: E402

import app  # noqa: E402

REPS = 3
CORES = os.cpu_count() or 1


def pcap_stats(pcap: Path) -> tuple[int, float]:
    n, first, last = 0, None, None
    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            t = float(pkt.time)
            first = t if first is None else first
            last = t
            n += 1
    return n, (last - first) if n else 0.0


def extract(pcap: Path, csv: Path) -> dict:
    """Ejecuta el extractor como lo hace el detector y mide con /usr/bin/time -l."""
    csv.unlink(missing_ok=True)
    cmd = ["/usr/bin/time", "-l", str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/convertir_pcap.py"), str(pcap), str(csv)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    err = res.stderr
    real = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", err)
    rss = re.search(r"(\d+)\s+maximum resident set size", err)
    cpu = float(real.group(2)) + float(real.group(3)) if real else None
    return {"s": wall, "cpu_s": cpu, "ram_mb": int(rss.group(1)) / 1e6 if rss else None}


def infer(csv: Path) -> dict:
    """Lectura + normalizacion + modelo + regla por tasa + decision por flujo (codigo de app.py)."""
    w0, c0 = time.perf_counter(), time.process_time()
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    df = app.normalize_cicflowmeter_columns(df)
    app.process_xgboost_file(csv, df)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    return {"s": wall, "cpu_s": cpu, "flujos": int(len(df)), "ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}


def med(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return st.median(values) if values else None


def main() -> None:
    out: dict = {"repeticiones": REPS, "nucleos_logicos": CORES, "capturas": {}}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        pcap = Path(path)
        packets, seconds = pcap_stats(pcap)
        csv = TMP / f"{name}.csv"
        ex = [extract(pcap, csv) for _ in range(REPS)]
        inf = [infer(csv) for _ in range(REPS)]
        e_s, e_cpu, i_s, i_cpu = med([x["s"] for x in ex]), med([x["cpu_s"] for x in ex]), med([x["s"] for x in inf]), med([x["cpu_s"] for x in inf])
        row = {
            "paquetes": packets, "tamano_mb": pcap.stat().st_size / 1e6, "ventana_s": seconds, "flujos": inf[0]["flujos"],
            "extraccion_s": e_s, "extraccion_cpu_s": e_cpu, "extraccion_cpu_pct_de_un_nucleo": 100 * e_cpu / e_s if e_cpu else None,
            "extraccion_ram_mb": med([x["ram_mb"] for x in ex]), "extraccion_paquetes_por_s": packets / e_s,
            "inferencia_s": i_s, "inferencia_cpu_s": i_cpu, "inferencia_cpu_pct_de_un_nucleo": 100 * i_cpu / i_s,
            "inferencia_ram_mb": med([x["ram_mb"] for x in inf]), "inferencia_ms_por_flujo": 1000 * i_s / max(inf[0]["flujos"], 1),
        }
        row["total_tras_la_captura_s"] = e_s + i_s
        row["total_con_ventana_s"] = seconds + e_s + i_s
        out["capturas"][name] = row
        print(f"{name:22s} paquetes {packets:6d} | ventana {seconds:5.1f} s | extraccion {e_s:6.2f} s (CPU {row['extraccion_cpu_pct_de_un_nucleo']:.0f} %, "
              f"{row['extraccion_ram_mb']:.0f} MB, {row['extraccion_paquetes_por_s']:.0f} paq/s) | inferencia {i_s * 1000:7.1f} ms "
              f"(CPU {row['inferencia_cpu_pct_de_un_nucleo']:.0f} %, {row['inferencia_ram_mb']:.0f} MB, {row['flujos']} flujos)")

    log = ROOT / "experimentos/resultados/mitigacion_bitacora.jsonl"
    fw = [json.loads(l).get("firewall_ms") for l in log.read_text().splitlines() if l.strip()]
    fw = [v for v in fw if v is not None]
    out["regla_firewall_ms"] = {"n": len(fw), "mediana": st.median(fw), "min": min(fw), "max": max(fw), "fuente": "bitacora del detector (pf real)"}
    print("insercion de la regla en pf:", out["regla_firewall_ms"])

    carga = json.loads((ROOT / "experimentos/resultados/carga_rf.json").read_text())["lotes"]
    out["carga_rf_cpu"] = [{"flujos_por_lote": r["flujos_por_lote"], "cpu_pct_de_un_nucleo": r["cpu_pct_de_un_nucleo"],
                            "cpu_pct_del_equipo": r["cpu_pct_del_equipo"], "ram_mb": r["ram_mb"]} for r in carga]
    (ROOT / "experimentos/resultados/medicion_cadena.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("CPU de la prueba de carga del RF (% del equipo):", {r["flujos_por_lote"]: round(r["cpu_pct_del_equipo"], 1) for r in carga})


if __name__ == "__main__":
    main()
