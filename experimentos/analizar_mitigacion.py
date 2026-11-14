"""Analiza los flujos capturados en la prueba de mitigacion con la misma logica de decision de app.py."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402  (solo se usan sus funciones; no arranca el servidor)

mit = json.loads((ROOT / "experimentos/resultados/mitigacion.json").read_text())
attacker = mit["ip_atacante"]
rows = []
for csv in sorted((ROOT / "data/flows").glob("flows-*.csv")):
    if csv.stat().st_size == 0:
        continue
    raw = pd.read_csv(csv)
    raw.columns = raw.columns.str.strip()
    if raw.empty:
        continue
    df = app.normalize_cicflowmeter_columns(raw)
    proba = app.model.predict_proba(app.prepare_xgboost_features(df))
    rates = app.source_rate_stats(df)
    labels = np.array(app.class_labels)
    p_attack = 1.0 - proba[:, list(labels).index("Benign")]
    for rec, p, pr in zip(df.to_dict(orient="records"), p_attack, proba):
        src = str(app.pick_value(rec, app.IP_COLUMNS, "?")); dst = str(app.pick_value(rec, app.DST_COLUMNS, "?"))
        port = app.pick_value(rec, app.PORT_COLUMNS, 0)
        pps = app.safe_number(rec.get("Flow Packets/s", 0)); bps = app.safe_number(rec.get("Flow Bytes/s", 0))
        rate = app.is_rate_attack_source(rates.get(src, {"pps": 0.0, "fps": 0.0}))
        noise = app.is_local_noise_flow(rec, dst, port, pps, bps)
        model_hit = p >= app.PREDICTION_THRESHOLD and pps >= app.MODEL_ATTACK_MIN_PPS
        block = (model_hit or rate) and not noise
        rows.append({"archivo": csv.name, "src": src, "dst": dst, "puerto": int(app.safe_number(port)), "pps": pps,
                     "p_ataque": float(p), "clase": str(labels[pr.argmax()]), "regla_tasa": rate, "modelo": bool(model_hit),
                     "ruido_local": noise, "bloquear": bool(block),
                     "paquetes": app.safe_number(rec.get("Total Fwd Packets", 0)) + app.safe_number(rec.get("Total Backward Packets", 0)),
                     "duracion_us": app.safe_number(rec.get("Flow Duration", 0))})
t = pd.DataFrame(rows)
events = [json.loads(l) for l in (ROOT / "data/state/eventos_mitigacion.jsonl").read_text().splitlines() if l.strip()]
attack_event = next(e for e in events if e["evento"] == "bloqueo" and e["t_deteccion_epoch"] >= mit["t_inicio_ataque"])
attack_file = attack_event["source_file"]
to_victim = t[t["puerto"] == 8080]
groups = {
    "legitimo (127.0.0.1 -> victima)": to_victim[to_victim["src"] == "127.0.0.1"],
    "ataque (IP local -> victima, ventana del ataque)": to_victim[(to_victim["src"] == attacker) & (to_victim["archivo"] == attack_file)],
    "sondeos desde la IP atacante fuera del ataque": to_victim[(to_victim["src"] == attacker) & (to_victim["archivo"] != attack_file)],
    "otro trafico de la maquina (no dirigido a la victima)": t[t["puerto"] != 8080],
}
out = {"flujos_totales": int(len(t)), "archivo_del_ataque": attack_file, "grupos": {}}
for name, g in groups.items():
    if g.empty:
        out["grupos"][name] = {"flujos": 0}; continue
    out["grupos"][name] = {
        "flujos": int(len(g)), "marcados_bloquear": int(g["bloquear"].sum()), "fraccion_bloquear": float(g["bloquear"].mean()),
        "por_modelo(p>=umbral)": int(g["modelo"].sum()), "por_regla_de_tasa": int(g["regla_tasa"].sum()),
        "solo_por_regla_de_tasa": int((g["regla_tasa"] & ~g["modelo"] & g["bloquear"]).sum()),
        "ruido_local_ignorado": int(g["ruido_local"].sum()),
        "p_ataque_mediana": float(g["p_ataque"].median()), "p_ataque_p90": float(g["p_ataque"].quantile(0.9)),
        "pps_mediana": float(g["pps"].median()), "paquetes_mediana": float(g["paquetes"].median()),
        "duracion_ms_mediana": float(g["duracion_us"].median() / 1000),
        "clases_predichas": g["clase"].value_counts().head(4).to_dict(),
    }
(ROOT / "experimentos/resultados/mitigacion_flujos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
t.to_csv(ROOT / "experimentos/resultados/mitigacion_flujos.csv", index=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
