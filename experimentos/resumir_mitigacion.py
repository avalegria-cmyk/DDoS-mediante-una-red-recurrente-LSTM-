"""Resume las repeticiones de la prueba de mitigacion (HTTP y UDP) y los falsos positivos por flujo."""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

RES = ROOT / "experimentos" / "resultados"
events = [json.loads(l) for l in (ROOT / "data/state/eventos_mitigacion.jsonl").read_text().splitlines() if l.strip()]
blocks = [e for e in events if e["evento"] == "bloqueo"]
expiries = [e["duracion_s"] for e in events if e["evento"] == "caducidad"]
out: dict = {}
for kind in ("http", "udp"):
    runs = []
    for f in sorted(glob.glob(str(RES / f"mitigacion_{kind}_rep*.json"))):
        d = json.loads(Path(f).read_text()); t0 = d["t_inicio_ataque"]
        b = next((e for e in blocks if t0 <= e["t_deteccion_epoch"] <= t0 + 150), None)
        runs.append({
            "corrida": Path(f).name, "bloqueado": b is not None,
            "s_inicio_ataque_a_regla": None if b is None else round(b["t_bloqueo_epoch"] - t0, 2),
            "disparo": None if b is None else b.get("disparo"), "prob_modelo": None if b is None else round(b["probability"], 3),
            "clase": None if b is None else b.get("attack"), "firewall_ms": None if b is None else b["firewall_ms"],
            "origen_pps": None if b is None else round(b.get("source_pps", 0), 1),
            "origen_flujos_s": None if b is None else round(b.get("source_flows_per_s", 0), 1),
            "s_bloqueo_efectivo": d.get("s_bloqueo_efectivo"), "bloqueos_en_fase_legitima": len(d.get("bloqueos_tras_fase_legitima", [])) if kind == "http" else None,
            "legitimo": d.get("legitimo_total"), "ataque_peticiones": d.get("ataque_peticiones"), "ataque_pps_udp": d.get("pps"),
        })
    t = [r["s_inicio_ataque_a_regla"] for r in runs if r["bloqueado"]]
    out[kind] = {"corridas": len(runs), "bloqueadas": len(t), "s_mediana": st.median(t) if t else None, "s_min": min(t) if t else None,
                 "s_max": max(t) if t else None, "firewall_ms_mediana": st.median([r["firewall_ms"] for r in runs if r["bloqueado"]]) if t else None,
                 "detalle": runs}
    if kind == "http":
        ok = sum(r["legitimo"]["ok"] for r in runs); bad = sum(r["legitimo"]["fallo"] for r in runs)
        out[kind]["legitimo_respondidas"] = ok; out[kind]["legitimo_fallidas"] = bad
        out[kind]["peticiones_ataque_por_s"] = round(st.mean(r["ataque_peticiones"]["ok"] for r in runs) / 20)
out["duracion_bloqueo_s"] = {"n": len(expiries), "min": min(expiries), "max": max(expiries), "configurado": 60}
out["errores_firewall"] = sum(1 for e in events if e.get("firewall_ok") is False)

# Falsos positivos por flujo en ventanas sin ataque (ventana completa de CAPTURE_SECONDS).
attacker = json.loads(Path(sorted(glob.glob(str(RES / "mitigacion_http_rep*.json")))[0]).read_text())["ip_atacante"]
legit = flagged = by_model = by_rate = windows = 0
for csv in sorted((ROOT / "data/flows").glob("flows-*.csv")):
    if csv.stat().st_size == 0:
        continue
    df = pd.read_csv(csv); df.columns = df.columns.str.strip()
    if df.empty or len(df) > 300:  # las ventanas con inundacion se excluyen
        continue
    df = app.normalize_cicflowmeter_columns(df)
    proba = app.model.predict_proba(app.prepare_xgboost_features(df))
    p_attack = 1.0 - proba[:, list(app.class_labels).index("Benign")]
    rates = app.source_rate_stats(df, app.CAPTURE_SECONDS)
    windows += 1
    for rec, p in zip(df.to_dict(orient="records"), p_attack):
        src = str(app.pick_value(rec, app.IP_COLUMNS, "?")); port = int(app.safe_number(app.pick_value(rec, app.PORT_COLUMNS, 0)))
        if src != "127.0.0.1" or port != 8080:
            continue
        pps = app.safe_number(rec.get("Flow Packets/s", 0))
        m = p >= app.PREDICTION_THRESHOLD and pps >= app.MODEL_ATTACK_MIN_PPS
        r = app.is_rate_attack_source(rates.get(src, {"pps": 0.0, "fps": 0.0}))
        legit += 1; by_model += bool(m); by_rate += bool(r); flagged += bool(m or r)
out["falsos_positivos_por_flujo"] = {"ventanas_sin_ataque": windows, "flujos_legitimos": legit, "marcados": flagged,
                                     "por_modelo": by_model, "por_regla_de_tasa": by_rate,
                                     "tasa": flagged / legit if legit else None, "ip_atacante": attacker}
(RES / "mitigacion_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({k: ({a: b for a, b in v.items() if a != "detalle"} if isinstance(v, dict) else v) for k, v in out.items()}, indent=1, ensure_ascii=False))
for k in ("http", "udp"):
    for r in out[k]["detalle"]:
        print(k, r["corrida"], r["s_inicio_ataque_a_regla"], r["disparo"], r["prob_modelo"], r["origen_pps"], r["origen_flujos_s"], r["firewall_ms"], r["s_bloqueo_efectivo"], r["legitimo"])
