"""Comparacion de politicas de decision sobre las capturas de la prueba de mitigacion.

Linea base sin aprendizaje (solo regla por tasa) frente a solo modelo y a la politica combinada
que usa el prototipo. La unidad es (ventana de captura, IP de origen): es lo que el sistema bloquea.

Uso: .venv/bin/python experimentos/ablacion_politicas.py CARPETA_CAPTURAS [CARPETA_CAPTURAS ...]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from scapy.utils import PcapReader

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

VICTIM_PORT = 8080


def pcap_seconds(pcap: Path) -> float | None:
    """Duracion real de la ventana: primer y ultimo paquete del PCAP."""
    if not pcap.exists() or pcap.stat().st_size <= 24:
        return None
    first = last = None
    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            t = float(pkt.time)
            first = t if first is None else first
            last = t
    return None if first is None else max(last - first, 0.5)


def main() -> None:
    folders = [Path(a) for a in sys.argv[1:]]
    attacker = sys.argv and None
    rows = []
    for folder in folders:
        for csv in sorted(folder.glob("flows-*.csv")):
            if csv.stat().st_size == 0:
                continue
            raw = pd.read_csv(csv)
            raw.columns = raw.columns.str.strip()
            if raw.empty:
                continue
            df = app.normalize_cicflowmeter_columns(raw)
            stamp = csv.name.removeprefix("flows-").removesuffix(".csv")
            window = pcap_seconds(folder / f"captura-{stamp}.pcap") or float(app.CAPTURE_SECONDS)
            window = min(max(window, 0.5), float(app.CAPTURE_SECONDS))
            proba = app.model.predict_proba(app.prepare_xgboost_features(df))
            p_attack = 1.0 - proba[:, list(app.class_labels).index("Benign")]
            rates = app.source_rate_stats(df, window)
            src_col = next(c for c in app.IP_COLUMNS if c in df.columns)
            port_col = next(c for c in app.PORT_COLUMNS if c in df.columns)
            dst_col = next((c for c in app.DST_COLUMNS if c in df.columns), None)
            recs = df.to_dict(orient="records")
            noise = [app.is_local_noise_flow(r, r.get(dst_col, ""), r.get(port_col, 0), app.safe_number(r.get("Flow Packets/s", 0)),
                                             app.safe_number(r.get("Flow Bytes/s", 0))) for r in recs]
            work = pd.DataFrame({
                "noise": noise,
                "src": df[src_col].astype(str), "port": pd.to_numeric(df[port_col], errors="coerce").fillna(0).astype(int),
                "p": p_attack, "pps": pd.to_numeric(df["Flow Packets/s"], errors="coerce").fillna(0),
                "fwd": pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0),
            })
            for src, g in work.groupby("src"):
                to_victim = g[g["port"] == VICTIM_PORT]
                rate = rates.get(src, {"pps": 0.0, "fps": 0.0})
                rows.append({
                    "carpeta": folder.name, "ventana": stamp, "ventana_s": round(window, 2), "src": src,
                    "flujos": int(len(g)), "flujos_a_victima": int(len(to_victim)), "paquetes_enviados": float(g["fwd"].sum()),
                    "src_pps": rate["pps"], "src_fps": rate["fps"], "p_max": float(g["p"].max()),
                    "modelo": bool(((g["p"] >= app.PREDICTION_THRESHOLD) & (g["pps"] >= app.MODEL_ATTACK_MIN_PPS) & ~g["noise"]).any()),
                    "max_paquetes_un_flujo": float(g["fwd"].max()),
                    "tasa": bool(app.is_rate_attack_source(rate)),
                })
    t = pd.DataFrame(rows)
    t["combinada"] = t["modelo"] | t["tasa"]

    # Verdad de terreno: el atacante es la IP local no-loopback; una ventana es de ataque para esa IP
    # si envia una inundacion (un flujo de mas de 300 paquetes o mas de 150 flujos hacia la victima en la ventana).
    lan = t[~t["src"].str.startswith("127.") & (t["flujos_a_victima"] > 0)]["src"].mode()
    attacker = lan.iloc[0] if len(lan) else ""
    t["es_ataque"] = (t["src"] == attacker) & ((t["max_paquetes_un_flujo"] > 300) | (t["flujos_a_victima"] > 150))
    t["tipo_ataque"] = ""
    t.loc[t["es_ataque"] & (t["flujos_a_victima"] > 150), "tipo_ataque"] = "HTTP"
    t.loc[t["es_ataque"] & (t["flujos_a_victima"] <= 150), "tipo_ataque"] = "UDP"
    t["grupo"] = "fondo (otro trafico de la maquina)"
    t.loc[(t["flujos_a_victima"] > 0) & ~t["es_ataque"], "grupo"] = "cliente legitimo de la victima"
    t.loc[t["es_ataque"], "grupo"] = "atacante"

    out: dict = {"ip_atacante": attacker, "unidad": "(ventana de captura, IP de origen)", "politicas": {}}
    benign = t[~t["es_ataque"]]
    for pol, label in (("tasa", "solo regla por tasa (linea base sin ML)"), ("modelo", "solo modelo RF-11"), ("combinada", "modelo + regla por tasa (prototipo)")):
        entry = {"descripcion": label}
        for kind in ("UDP", "HTTP"):
            a = t[t["tipo_ataque"] == kind]
            entry[f"ataque_{kind}_ventanas"] = int(len(a)); entry[f"ataque_{kind}_detectadas"] = int(a[pol].sum())
        entry["benignas_unidades"] = int(len(benign)); entry["benignas_marcadas"] = int(benign[pol].sum())
        for g in ("cliente legitimo de la victima", "fondo (otro trafico de la maquina)"):
            b = benign[benign["grupo"] == g]
            entry[f"marcadas | {g}"] = f"{int(b[pol].sum())} / {len(b)}"
        out["politicas"][pol] = entry
    res = ROOT / "experimentos" / "resultados"
    (res / "ablacion_politicas.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    t.to_csv(res / "ablacion_politicas.csv", index=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    fp = benign[benign["combinada"]]
    if len(fp):
        print("\nFalsos positivos de la politica combinada:")
        print(fp[["carpeta", "ventana", "src", "flujos", "src_pps", "src_fps", "p_max", "modelo", "tasa", "grupo"]].to_string(index=False))


if __name__ == "__main__":
    main()
