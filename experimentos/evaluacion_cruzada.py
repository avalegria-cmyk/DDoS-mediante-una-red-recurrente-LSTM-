"""Generalizacion a un segundo dataset (correccion pedida por los revisores 1 y 3).

Entrena con CIC-DDoS2019 y evalua, SIN reentrenar ni reajustar, sobre otro dataset con las
mismas 77 variables de CICFlowMeter (p. ej. CICIDS2017 en Parquet). Mismo protocolo que
evaluacion_unificada.py: tarea binaria, instancia = ultimo flujo de cada ventana de 10 flujos,
imputador y escalador ajustados solo con el entrenamiento, varias semillas.

Uso:
    python3 experimentos/evaluacion_cruzada.py --test-dir RUTA/cicids2017 [--test-files DDoS DoS Benign]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluacion_unificada as eu  # noqa: E402

MODELS = ["rf11_multi", "rf11_bin", "rf77", "xgb77", "mlp77", "mlp47", "mlp11", "lstm77", "gru77"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, default=eu.DEFAULT_ARCHIVE_DIR, help="CIC-DDoS2019 en Parquet")
    ap.add_argument("--test-dir", type=Path, required=True, help="Segundo dataset en Parquet")
    ap.add_argument("--test-files", nargs="*", default=["DDoS", "DoS", "Benign"],
                    help="Solo se usan los archivos cuyo nombre contiene alguno de estos textos")
    ap.add_argument("--train-files", choices=["todos", "training"], default="todos",
                    help="todos: los 17 archivos de CIC-DDoS2019 (todos los tipos de ataque); training: solo *-training")
    ap.add_argument("--max-rows-train", type=int, default=12000)
    ap.add_argument("--max-rows-test", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=eu.ALL_MODELS)
    ap.add_argument("--tag", default="cicids2017")
    ap.add_argument("--output-dir", type=Path, default=eu.DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    train_paths = sorted(args.train_dir.expanduser().glob("*-training.parquet" if args.train_files == "training" else "*.parquet"))
    test_paths = [p for p in sorted(args.test_dir.expanduser().glob("*.parquet"))
                  if not args.test_files or any(k.lower() in p.name.lower() for k in args.test_files)]
    if not train_paths or not test_paths:
        raise FileNotFoundError(f"Sin archivos: train={len(train_paths)} test={len(test_paths)}")

    print("--- ENTRENAMIENTO (CIC-DDoS2019) ---")
    train_df = pd.concat([eu.read_file(p, args.max_rows_train) for p in train_paths], ignore_index=True)
    print(f"  {len(train_paths)} archivos, {len(train_df)} flujos")
    print("--- PRUEBA (segundo dataset) ---")
    test_parts, inventory = [], []
    for p in test_paths:
        df = eu.read_file(p, args.max_rows_test)
        labels = df["Label"].map(eu.normalize_label).value_counts().to_dict()
        inventory.append({"archivo": p.name, "filas_usadas": int(len(df)), "etiquetas": labels})
        print(f"  {p.name:40s} {len(df):7d} filas  {labels}")
        test_parts.append(df)
    test_df = pd.concat(test_parts, ignore_index=True)

    feature_cols = [c for c in train_df.columns if c not in {"Label", "source_file"} and pd.api.types.is_numeric_dtype(train_df[c])]
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise KeyError(f"El segundo dataset no tiene estas columnas: {missing}")
    feats47 = json.loads((eu.ROOT / "modelo" / "feature_cols_lstm_gru.json").read_text(encoding="utf-8"))
    col_pos = {c: i for i, c in enumerate(feature_cols)}

    def to_matrix(df: pd.DataFrame) -> np.ndarray:
        return df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float64")

    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(to_matrix(train_df))).astype("float32")
    X_test = scaler.transform(imputer.transform(to_matrix(test_df))).astype("float32")
    y_train_multi = train_df["Label"].map(eu.normalize_label).to_numpy()
    y_test_multi = test_df["Label"].map(eu.normalize_label).to_numpy()
    y_train, y_test = (y_train_multi != "Benign").astype(int), (y_test_multi != "Benign").astype(int)

    win_train = eu.window_index(train_df["source_file"].to_numpy(), 10, 2)
    win_test = eu.window_index(test_df["source_file"].to_numpy(), 10, 2)
    rows = win_test[:, -1]
    y_eval, y_eval_multi, file_eval = y_test[rows], y_test_multi[rows], test_df["source_file"].to_numpy()[rows]
    d = {"X_train": X_train, "X_test": X_test, "X_eval": X_test[rows], "y_train": y_train, "y_train_multi": y_train_multi,
         "win_train": win_train, "win_test": win_test, "idx77": list(range(len(feature_cols))),
         "idx47": [col_pos[c] for c in feats47], "idx25": [], "idx11": [col_pos[c] for c in eu.FEATURES_11]}
    print(f"\nInstancias de prueba: {len(y_eval)} (Benign {int((y_eval == 0).sum())}, ataque {int((y_eval == 1).sum())})")

    results: dict = {}
    for key in args.models:
        runs, first = [], None
        for rep in range(args.seeds):
            seed = eu.DATA_SEED + rep
            if key[:3] in {"lst", "gru"}:
                pred, score, *_ = eu.run_rnn(key, seed, d, args.epochs)
            else:
                pred, score, *_ = eu.run_tabular(key, seed, d)
            m = eu.binary_metrics(y_eval, pred, score)
            m.update(seed=seed, train_seconds=0.0, ms_por_instancia=0.0)
            runs.append(m)
            if rep == 0:
                first = pred
            print(f"  {key:11s} semilla {seed}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
                  f"f1={m['f1']:.4f} auc={m['auc_roc']:.4f} fpr={m['fpr']:.4f}")
        # Deteccion por tipo de ataque y falsos positivos por archivo (primera semilla).
        by_label = {}
        for label in sorted(set(y_eval_multi)):
            mask = y_eval_multi == label
            by_label[label] = {"instancias": int(mask.sum()), "marcadas_como_ataque": float(first[mask].mean())}
        by_file = {}
        for name in pd.unique(file_eval):
            mask = (file_eval == name) & (y_eval == 0)
            if mask.any():
                by_file[name] = {"benignas": int(mask.sum()), "fpr": float(first[mask].mean())}
        results[key] = {"nombre": eu.MODEL_NAMES[key], "repeticiones": runs, "resumen": eu.summarize(runs),
                        "por_etiqueta": by_label, "fpr_por_archivo": by_file}

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocolo": {"entrenamiento": f"CIC-DDoS2019 ({args.train_files})", "prueba": str(args.test_dir.name), "tarea": "binaria",
                      "sin_reentrenar": True, "semillas": args.seeds, "max_rows_train": args.max_rows_train, "max_rows_test": args.max_rows_test},
        "datos": {"archivos_prueba": inventory, "filas_train": int(len(train_df)), "instancias_prueba": int(len(y_eval)),
                  "benignas": int((y_eval == 0).sum()), "ataque": int((y_eval == 1).sum()),
                  "etiquetas_prueba": pd.Series(y_eval_multi).value_counts().to_dict()},
        "resultados": results,
    }
    (out_dir / f"resultados_cruzada_{args.tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n--- GENERALIZACION A {args.test_dir.name} ({len(y_eval)} instancias, {args.seeds} semillas) ---")
    for key in sorted(results, key=lambda k: -results[k]["resumen"]["f1"]["mean"]):
        r = results[key]["resumen"]
        print(f"{key:11s} acc {eu.fmt(r['accuracy'])} | prec {eu.fmt(r['precision'])} | rec {eu.fmt(r['recall'])} | "
              f"f1 {eu.fmt(r['f1'])} | auc {eu.fmt(r['auc_roc'])} | fpr {eu.fmt(r['fpr'])}")
    if "rf11_multi" in results:
        print("\nModelo desplegado: fraccion marcada como ataque por etiqueta")
        for label, v in results["rf11_multi"]["por_etiqueta"].items():
            print(f"  {label:28s} {v['instancias']:7d}  {v['marcadas_como_ataque']:.4f}")
    print(f"[OK] {out_dir / f'resultados_cruzada_{args.tag}.json'}")


if __name__ == "__main__":
    main()
