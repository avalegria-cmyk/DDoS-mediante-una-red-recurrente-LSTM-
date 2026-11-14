"""Compara CICFlowMeter Java (oficial, el que genero CIC-DDoS2019) con cicflowmeter 0.5.0 (Python, el desplegado).

Ambas herramientas procesan los mismos PCAP. Los flujos se emparejan por la 5-tupla y se comparan
las 11 variables del modelo, asi como la probabilidad de ataque que da el modelo con cada version.
Uso: .venv/bin/python experimentos/comparar_extractores.py CARPETA   (con subcarpetas java/ y python/)
"""
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
import app  # noqa: E402

JAVA_NAMES = {
    "Total Fwd Packet": "Total Fwd Packets", "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Fwd Packets Length Total", "Total Length of Bwd Packet": "Bwd Packets Length Total",
}
FEATURES = list(app.feature_cols)
KEY = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]


def load_java(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df.rename(columns=JAVA_NAMES)


def load_python(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = app.normalize_cicflowmeter_columns(df)
    return df.rename(columns={"src_ip": "Src IP", "dst_ip": "Dst IP", "src_port": "Src Port", "dst_port": "Dst Port", "protocol": "Protocol"})


def p_attack(df: pd.DataFrame) -> np.ndarray:
    proba = app.model.predict_proba(app.prepare_xgboost_features(df[FEATURES].reset_index(drop=True)))
    return 1.0 - proba[:, list(app.class_labels).index("Benign")]


def main() -> None:
    base = Path(sys.argv[1])
    out: dict = {"capturas": {}, "variables": {}}
    pairs = []
    for jfile in sorted((base / "java").glob("*_Flow.csv")):
        name = jfile.name.replace(".pcap_Flow.csv", "")
        java, py = load_java(jfile), load_python(base / "python" / f"{name}.csv")
        for d in (java, py):
            for c in ("Src Port", "Dst Port", "Protocol"):
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype(int)
            d["orden"] = d.groupby(KEY).cumcount()
        merged = java.merge(py, on=KEY + ["orden"], suffixes=("_java", "_py"))
        merged["captura"] = name
        out["capturas"][name] = {"flujos_java": int(len(java)), "flujos_python": int(len(py)), "emparejados": int(len(merged))}
        pairs.append(merged)
    m = pd.concat(pairs, ignore_index=True)

    for f in FEATURES:
        a = pd.to_numeric(m[f + "_java"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(m[f + "_py"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        rel = (b - a).abs() / np.maximum(a.abs(), 1e-9)
        rel[(a == 0) & (b == 0)] = 0.0
        out["variables"][f] = {
            "n": int(ok.sum()), "mediana_java": float(a.median()), "mediana_python": float(b.median()),
            "error_relativo_mediano": float(rel.median()), "error_relativo_p90": float(rel.quantile(0.9)),
            "fraccion_dentro_del_1pct": float((rel <= 0.01).mean()), "fraccion_dentro_del_10pct": float((rel <= 0.10).mean()),
            "razon_mediana_python_sobre_java": float((b[a != 0] / a[a != 0]).median()) if (a != 0).any() else None,
            "correlacion_spearman": float(a.corr(b, method="spearman")) if a.nunique() > 1 and b.nunique() > 1 else None,
        }

    pj = p_attack(m[[f + "_java" for f in FEATURES]].rename(columns=lambda c: c.removesuffix("_java")))
    pp = p_attack(m[[f + "_py" for f in FEATURES]].rename(columns=lambda c: c.removesuffix("_py")))
    thr = app.PREDICTION_THRESHOLD
    out["modelo"] = {
        "flujos": int(len(m)), "diferencia_absoluta_media_de_probabilidad": float(np.abs(pj - pp).mean()),
        "diferencia_absoluta_p90": float(np.quantile(np.abs(pj - pp), 0.9)),
        "acuerdo_en_la_decision": float(((pj >= thr) == (pp >= thr)).mean()),
        "marcados_con_java": int((pj >= thr).sum()), "marcados_con_python": int((pp >= thr).sum()),
        "por_captura": {c: {"flujos": int((m["captura"] == c).sum()),
                            "p_media_java": float(pj[(m["captura"] == c).to_numpy()].mean()),
                            "p_media_python": float(pp[(m["captura"] == c).to_numpy()].mean()),
                            "acuerdo": float((((pj >= thr) == (pp >= thr))[(m["captura"] == c).to_numpy()]).mean())}
                        for c in m["captura"].unique()},
    }
    (ROOT / "experimentos/resultados/comparacion_extractores.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["capturas"], ensure_ascii=False))
    print(f"{'variable':28s} {'n':>5s} {'med_java':>12s} {'med_py':>12s} {'err_med':>8s} {'err_p90':>8s} {'<=1%':>6s} {'<=10%':>6s} {'razon':>8s} {'rho':>6s}")
    for f, v in out["variables"].items():
        print(f"{f:28s} {v['n']:5d} {v['mediana_java']:12.2f} {v['mediana_python']:12.2f} {v['error_relativo_mediano']:8.4f} "
              f"{v['error_relativo_p90']:8.4f} {v['fraccion_dentro_del_1pct']:6.2f} {v['fraccion_dentro_del_10pct']:6.2f} "
              f"{(v['razon_mediana_python_sobre_java'] or 0):8.3f} {(v['correlacion_spearman'] or 0):6.3f}")
    print(json.dumps(out["modelo"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
