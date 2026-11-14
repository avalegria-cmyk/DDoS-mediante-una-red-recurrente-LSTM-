"""Evaluacion unificada de todos los modelos del articulo (correcciones A y E).

Responde a los revisores 1 y 3: todos los modelos se entrenan con la MISMA
particion de entrenamiento y se evaluan sobre las MISMAS instancias de prueba,
con la MISMA tarea (binaria: Benign vs. DDoS).

Protocolo
---------
* Unidad de evaluacion: el ultimo flujo de cada ventana de `timesteps` flujos
  consecutivos (dentro de un mismo archivo Parquet), con salto `stride`.
* Etiqueta de la instancia: la etiqueta de ese ultimo flujo.
* Modelos secuenciales (LSTM/GRU): reciben la ventana completa.
* Modelos tabulares (RF, XGBoost, MLP, Isolation Forest): reciben solo ese
  ultimo flujo. Asi, las instancias y las etiquetas son identicas para todos.
* El imputador y el escalador se ajustan unicamente con entrenamiento.
* Cada modelo se repite con varias semillas; se reporta media, desviacion
  estandar e intervalo de confianza del 95 %, ademas de AUC-ROC.
* Prueba de McNemar entre el modelo desplegado y cada uno de los demas.

Uso
---
    python3 experimentos/evaluacion_unificada.py --archive-dir RUTA/archive
    python3 experimentos/evaluacion_unificada.py --archive-dir RUTA/archive --split bloques
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = ROOT.parent / "archive"
DEFAULT_OUTPUT_DIR = ROOT / "experimentos" / "resultados"
DATA_SEED = 42  # semilla fija del muestreo de filas: los datos no cambian entre repeticiones

# Las 11 variables del modelo compacto desplegado (scripts/entrenar_rf_compacto_ddos.py).
FEATURES_11 = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Min",
    "Flow IAT Max",
]

# Hiperparametros: los del RF compacto y LSTM/GRU son los del repositorio.
# Los de XGBoost, MLP e Isolation Forest no estaban guardados en el repositorio;
# se fijan aqui y deben reportarse tal cual en el articulo.
RF_PARAMS = dict(n_estimators=180, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, eval_metric="logloss",
)
MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=0.001,
    batch_size=128, max_iter=50, early_stopping=True, validation_fraction=0.2, n_iter_no_change=3,
)
IFOREST_PARAMS = dict(n_estimators=200, max_samples=256, contamination="auto", n_jobs=-1)
IFOREST_BENIGN_PERCENTILE = 95.0  # umbral: percentil 95 de la puntuacion de anomalia en Benign de entrenamiento
RNN_PARAMS = dict(units=64, dropout_1=0.30, dense=32, dropout_2=0.20, learning_rate=0.001, batch_size=128, patience=3)

ALL_MODELS = [
    "rf11_multi", "rf11_bin", "rf77", "xgb77", "mlp77", "mlp47", "mlp11", "iforest77",
    "lstm77", "gru77", "lstm47", "lstm25", "gru25",
]
MODEL_NAMES = {
    "rf11_multi": "Compact Random Forest (11 features, multiclass grouped, deployed)",
    "rf11_bin": "Compact Random Forest (11 features, binary)",
    "rf77": "Random Forest (77 features)",
    "xgb77": "XGBoost (77 features)",
    "mlp77": "MLP (77 features)",
    "mlp47": "MLP (47 features)",
    "mlp11": "MLP (11 features)",
    "iforest77": "Isolation Forest (77 features)",
    "lstm77": "LSTM (77 features)",
    "gru77": "GRU (77 features)",
    "lstm47": "LSTM (47 features)",
    "lstm25": "LSTM (25 features)",
    "gru25": "GRU (25 features)",
}
REFERENCE_MODEL = "rf11_multi"


# --------------------------------------------------------------------------- datos

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR, help="Carpeta con los .parquet de CIC-DDoS2019")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--split", choices=["holdout", "bloques"], default="holdout",
        help="holdout: *-training.parquet para entrenar y *-testing.parquet para probar. "
             "bloques: primer 70 %% contiguo de cada archivo para entrenar y ultimo 30 %% para probar.",
    )
    p.add_argument("--seeds", type=int, default=5, help="Numero de repeticiones por modelo")
    p.add_argument("--max-rows-per-file", type=int, default=12000)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--tag", default="", help="Sufijo para los archivos de salida (evita sobrescribir una corrida completa)")
    return p.parse_args()


def normalize_label(value: object) -> str:
    label = str(value).strip()
    if label.lower() == "benign":
        return "Benign"
    label = label.replace("DrDoS_", "")
    return {"UDP-lag": "UDPLag", "UDP_Lag": "UDPLag"}.get(label, label)


def read_file(path: Path, max_rows: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        # Igual que en el cuaderno original: se muestrea y se conserva el orden del archivo.
        df = df.sample(n=max_rows, random_state=DATA_SEED).sort_index()
    df = df.reset_index(drop=True)
    df["source_file"] = path.name
    return df


def load_data(archive_dir: Path, split: str, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    files = sorted(archive_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay archivos .parquet en {archive_dir}")
    train_parts, test_parts, inventory = [], [], []
    for path in files:
        df = read_file(path, max_rows)
        labels = df["Label"].map(normalize_label).value_counts().to_dict()
        if split == "holdout":
            if path.stem.endswith("-training"):
                train_parts.append(df); role = "train"
            elif path.stem.endswith("-testing"):
                test_parts.append(df); role = "test"
            else:
                role = "ignorado"
        else:
            cut = int(len(df) * 0.70)
            train_parts.append(df.iloc[:cut]); test_parts.append(df.iloc[cut:]); role = "train 70 % / test 30 %"
        inventory.append({"archivo": path.name, "filas_usadas": int(len(df)), "rol": role, "etiquetas": labels})
        print(f"  {path.name:28s} {len(df):7d} filas  [{role}]  {labels}")
    if not train_parts or not test_parts:
        raise FileNotFoundError("No se encontraron archivos de entrenamiento y de prueba para este split.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True), inventory


def window_index(source_files: np.ndarray, timesteps: int, stride: int) -> np.ndarray:
    """Matriz (n_ventanas, timesteps) con los indices de fila de cada ventana; nunca cruza archivos."""
    rows = []
    for name in pd.unique(source_files):
        idx = np.flatnonzero(source_files == name)
        if len(idx) < timesteps:
            continue
        starts = np.arange(0, len(idx) - timesteps + 1, stride)
        rows.append(idx[starts[:, None] + np.arange(timesteps)[None, :]])
    return np.concatenate(rows, axis=0)


def window_diagnostics(win: np.ndarray, y_bin: np.ndarray, y_multi: np.ndarray) -> dict:
    """Responde al revisor 3 (punto 4): ¿las ventanas mezclan etiquetas? ¿la tarea es trivial?"""
    wb = y_bin[win]
    mixed_bin = (wb.min(axis=1) != wb.max(axis=1))
    wm = y_multi[win]
    mixed_multi = np.array([len(set(r)) > 1 for r in wm])
    last = wb[:, -1]
    prev = wb[:, -2]
    majority_prev = (wb[:, :-1].mean(axis=1) >= 0.5).astype(int)
    return {
        "ventanas": int(len(win)),
        "ventanas_benignas": int((last == 0).sum()),
        "ventanas_ataque": int((last == 1).sum()),
        "fraccion_ventanas_mixtas_binario": float(mixed_bin.mean()),
        "fraccion_ventanas_mixtas_multiclase": float(mixed_multi.mean()),
        # Lineas base que usan las etiquetas reales de los flujos anteriores (no son modelos
        # desplegables): miden cuanta informacion da solo la autocorrelacion de etiquetas.
        "exactitud_copiando_etiqueta_flujo_anterior": float((prev == last).mean()),
        "exactitud_mayoria_de_los_flujos_previos": float((majority_prev == last).mean()),
    }


# --------------------------------------------------------------------------- metricas

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, score),
        "auc_pr": average_precision_score(y_true, score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "fpr": fp / max(fp + tn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def summarize(runs: list[dict]) -> dict:
    out = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "balanced_accuracy", "fpr",
                "train_seconds", "ms_por_instancia"]:
        values = np.array([r[key] for r in runs], dtype=float)
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        half = stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        out[key] = {"mean": float(mean), "std": float(std), "ci95_low": float(mean - half), "ci95_high": float(mean + half)}
    return out


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = 1.0 if only_a + only_b == 0 else float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {"solo_acierta_referencia": only_a, "solo_acierta_otro": only_b, "p_value": p}


# --------------------------------------------------------------------------- modelos

def run_tabular(key: str, seed: int, d: dict) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    cols = d["idx11"] if key.endswith("11") or key.startswith("rf11") else d["idx47"] if key.endswith("47") else d["idx77"]
    Xtr, Xte = d["X_train"][:, cols], d["X_eval"][:, cols]
    extra: dict = {}
    t0 = time.perf_counter()
    if key == "rf11_multi":
        model = RandomForestClassifier(random_state=seed, **RF_PARAMS).fit(Xtr, d["y_train_multi"])
    elif key in {"rf11_bin", "rf77"}:
        model = RandomForestClassifier(random_state=seed, **RF_PARAMS).fit(Xtr, d["y_train"])
    elif key == "xgb77":
        from xgboost import XGBClassifier
        pos = max(int(d["y_train"].sum()), 1)
        model = XGBClassifier(random_state=seed, scale_pos_weight=(len(d["y_train"]) - pos) / pos, **XGB_PARAMS)
        model.fit(Xtr, d["y_train"])
    elif key.startswith("mlp"):
        model = MLPClassifier(random_state=seed, **MLP_PARAMS).fit(Xtr, d["y_train"])
    elif key == "iforest77":
        benign = Xtr[d["y_train"] == 0]
        model = IsolationForest(random_state=seed, **IFOREST_PARAMS).fit(benign)
        threshold = float(np.percentile(-model.score_samples(benign), IFOREST_BENIGN_PERCENTILE))
        extra["umbral_anomalia"] = threshold
    else:
        raise ValueError(key)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if key == "rf11_multi":
        proba = model.predict_proba(Xte)
        classes = [str(c) for c in model.classes_]
        labels = np.array(classes)[proba.argmax(axis=1)]
        pred = (labels != "Benign").astype(int)
        score = 1.0 - proba[:, classes.index("Benign")]
        extra["pred_multiclase"] = labels
        extra["importancias"] = model.feature_importances_.tolist()
    elif key == "iforest77":
        score = -model.score_samples(Xte)
        pred = (score > threshold).astype(int)
    else:
        score = model.predict_proba(Xte)[:, 1]
        pred = (score >= 0.5).astype(int)
        if hasattr(model, "feature_importances_"):
            extra["importancias"] = model.feature_importances_.tolist()
    infer_s = time.perf_counter() - t0
    return pred, score, train_s, infer_s, extra


def run_rnn(key: str, seed: int, d: dict, epochs: int) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models

    tf.keras.utils.set_random_seed(seed)
    cols = {"77": d["idx77"], "47": d["idx47"], "25": d["idx25"]}[key[-2:]]
    Xtr = d["X_train"][d["win_train"]][:, :, cols]
    ytr = d["y_train"][d["win_train"][:, -1]]
    Xte = d["X_test"][d["win_test"]][:, :, cols]

    # Se barajan las ventanas para que el 20 % de validacion sea representativo de todos los archivos.
    order = np.random.default_rng(seed).permutation(len(Xtr))
    Xtr, ytr = Xtr[order], ytr[order]
    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    cell = layers.LSTM if key.startswith("lstm") else layers.GRU
    model = models.Sequential([
        layers.Input(shape=Xtr.shape[1:]),
        layers.Masking(mask_value=0.0),
        cell(RNN_PARAMS["units"]),
        layers.Dropout(RNN_PARAMS["dropout_1"]),
        layers.Dense(RNN_PARAMS["dense"], activation="relu"),
        layers.Dropout(RNN_PARAMS["dropout_2"]),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(RNN_PARAMS["learning_rate"]), loss="binary_crossentropy")
    t0 = time.perf_counter()
    history = model.fit(
        Xtr, ytr, validation_split=0.2, epochs=epochs, batch_size=RNN_PARAMS["batch_size"],
        class_weight=class_weight, verbose=0,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=RNN_PARAMS["patience"], restore_best_weights=True)],
    )
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = model.predict(Xte, batch_size=1024, verbose=0).ravel()
    infer_s = time.perf_counter() - t0
    tf.keras.backend.clear_session()
    return (score >= 0.5).astype(int), score, train_s, infer_s, {"epocas_entrenadas": len(history.history["loss"])}


# --------------------------------------------------------------------------- salida

def fmt(stat: dict, digits: int = 4) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def write_tables(results: dict, out_dir: Path, tag: str, n_eval: int) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        s = res["resumen"]
        rows.append({
            "modelo": MODEL_NAMES[key], "clave": key,
            **{f"{m}_mean": s[m]["mean"] for m in ["accuracy", "precision", "recall", "f1", "auc_roc", "fpr"]},
            **{f"{m}_std": s[m]["std"] for m in ["accuracy", "precision", "recall", "f1", "auc_roc", "fpr"]},
            "f1_ci95_low": s["f1"]["ci95_low"], "f1_ci95_high": s["f1"]["ci95_high"],
            "ms_por_instancia": s["ms_por_instancia"]["mean"],
            "mcnemar_p_vs_desplegado": res.get("mcnemar_vs_referencia", {}).get("p_value"),
        })
    table = pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    table.to_csv(out_dir / f"tabla_{tag}.csv", index=False)

    lines = [
        "% Generado por experimentos/evaluacion_unificada.py",
        r"\begin{tabular}{clccccc}", r"\toprule",
        r"\textbf{Rank} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} & \textbf{AUC-ROC}\\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        cells = " & ".join(f"{r[m + '_mean']:.4f} $\\pm$ {r[m + '_std']:.4f}" for m in ["accuracy", "precision", "recall", "f1", "auc_roc"])
        lines.append(f"{r['rank']} & {r['modelo']} & {cells}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", f"% Instancias de prueba (identicas para todos los modelos): {n_eval}"]
    (out_dir / f"tabla_{tag}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.split + (f"_{args.tag}" if args.tag else "")

    print(f"--- CARGANDO CIC-DDoS2019 desde {archive_dir} (split: {args.split}) ---")
    train_df, test_df, inventory = load_data(archive_dir, args.split, args.max_rows_per_file)

    feature_cols = [c for c in train_df.columns if c not in {"Label", "source_file"}
                    and pd.api.types.is_numeric_dtype(train_df[c])]
    feats47 = json.loads((ROOT / "modelo" / "feature_cols_lstm_gru.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(ROOT / "modelos_lstm_reales_47" / "ranking_caracteristicas_reales.csv")
    feats25 = ranking.sort_values("rank")["feature"].head(25).tolist()
    missing = sorted(set(FEATURES_11 + feats47 + feats25) - set(feature_cols))
    if missing:
        raise KeyError(f"Faltan columnas en el dataset: {missing}")
    col_pos = {c: i for i, c in enumerate(feature_cols)}

    def to_matrix(df: pd.DataFrame) -> np.ndarray:
        return df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float64")

    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(to_matrix(train_df))).astype("float32")
    X_test = scaler.transform(imputer.transform(to_matrix(test_df))).astype("float32")

    y_train_multi = train_df["Label"].map(normalize_label).to_numpy()
    y_test_multi = test_df["Label"].map(normalize_label).to_numpy()
    y_train = (y_train_multi != "Benign").astype(int)
    y_test = (y_test_multi != "Benign").astype(int)

    win_train = window_index(train_df["source_file"].to_numpy(), args.timesteps, args.stride)
    win_test = window_index(test_df["source_file"].to_numpy(), args.timesteps, args.stride)
    eval_rows = win_test[:, -1]
    y_eval, y_eval_multi = y_test[eval_rows], y_test_multi[eval_rows]

    d = {
        "X_train": X_train, "X_test": X_test, "X_eval": X_test[eval_rows],
        "y_train": y_train, "y_train_multi": y_train_multi,
        "win_train": win_train, "win_test": win_test,
        "idx77": list(range(len(feature_cols))),
        "idx47": [col_pos[c] for c in feats47],
        "idx25": [col_pos[c] for c in feats25],
        "idx11": [col_pos[c] for c in FEATURES_11],
    }

    diag_train = window_diagnostics(win_train, y_train, y_train_multi)
    diag_test = window_diagnostics(win_test, y_test, y_test_multi)
    print(f"\nVariables: {len(feature_cols)} | filas train: {len(train_df)} | filas test: {len(test_df)}")
    print(f"Ventanas train: {len(win_train)} | instancias de prueba comunes: {len(eval_rows)} "
          f"(Benign {int((y_eval == 0).sum())}, ataque {int((y_eval == 1).sum())})")
    print(f"Ventanas de prueba con etiquetas mezcladas: {diag_test['fraccion_ventanas_mixtas_binario']:.2%} | "
          f"exactitud copiando la etiqueta del flujo anterior: {diag_test['exactitud_copiando_etiqueta_flujo_anterior']:.4f}")

    results: dict = {}
    first_preds: dict = {}
    for key in args.models:
        runs, extras = [], {}
        for rep in range(args.seeds):
            seed = DATA_SEED + rep
            runner = run_rnn if key[:3] in {"lst", "gru"} else run_tabular
            pred, score, train_s, infer_s, extra = (
                runner(key, seed, d, args.epochs) if runner is run_rnn else runner(key, seed, d)
            )
            m = binary_metrics(y_eval, pred, score)
            m.update(seed=seed, train_seconds=train_s, ms_por_instancia=1000 * infer_s / len(y_eval))
            for k in ("umbral_anomalia", "epocas_entrenadas"):
                if k in extra:
                    m[k] = extra[k]
            runs.append(m)
            if rep == 0:
                first_preds[key] = pred
                extras = extra
            print(f"  {key:11s} semilla {seed}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                  f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc_roc']:.4f}")
        results[key] = {"nombre": MODEL_NAMES[key], "repeticiones": runs, "resumen": summarize(runs)}
        if "importancias" in extras:
            names = FEATURES_11 if key.startswith("rf11") else feature_cols
            imp = sorted(zip(names, extras["importancias"]), key=lambda t: -t[1])
            results[key]["importancia_variables"] = [{"variable": n, "importancia": float(v)} for n, v in imp]
        if key == "rf11_multi":
            labels = sorted(set(y_eval_multi) | set(extras["pred_multiclase"]), key=lambda s: (s != "Benign", s))
            results[key]["reporte_multiclase"] = classification_report(
                y_eval_multi, extras["pred_multiclase"], labels=labels, digits=4, zero_division=0)
            results[key]["reporte_multiclase_dict"] = classification_report(
                y_eval_multi, extras["pred_multiclase"], labels=labels, zero_division=0, output_dict=True)

    if REFERENCE_MODEL in first_preds:
        for key in results:
            if key != REFERENCE_MODEL:
                results[key]["mcnemar_vs_referencia"] = mcnemar(y_eval, first_preds[REFERENCE_MODEL], first_preds[key])

    # Posicion de las 11 variables en el ranking de importancia del RF de 77 variables.
    selection = None
    if "rf77" in results:
        order = [r["variable"] for r in results["rf77"]["importancia_variables"]]
        selection = [{"variable": f, "posicion_en_rf77": order.index(f) + 1,
                      "importancia_en_rf77": results["rf77"]["importancia_variables"][order.index(f)]["importancia"]}
                     for f in FEATURES_11]

    payload = {
        "protocolo": {
            "split": args.split, "tarea": "binaria (Benign=0, DDoS=1)",
            "unidad_de_evaluacion": "ultimo flujo de cada ventana; identica para todos los modelos",
            "timesteps": args.timesteps, "stride": args.stride, "max_rows_per_file": args.max_rows_per_file,
            "semillas": [DATA_SEED + i for i in range(args.seeds)], "epocas_max": args.epochs,
            "preprocesamiento": "inf->NaN, imputacion por mediana y StandardScaler ajustados solo en entrenamiento",
            "modelo_de_referencia_mcnemar": REFERENCE_MODEL,
        },
        "datos": {
            "archive_dir": str(archive_dir), "archivos": inventory,
            "n_variables": len(feature_cols), "variables_77": feature_cols,
            "variables_47": feats47, "variables_25": feats25, "variables_11": FEATURES_11,
            "filas_train": int(len(train_df)), "filas_test": int(len(test_df)),
            "clases_train": pd.Series(y_train_multi).value_counts().to_dict(),
            "clases_test_evaluadas": pd.Series(y_eval_multi).value_counts().to_dict(),
            "ventanas_train": diag_train, "ventanas_test": diag_test,
        },
        "hiperparametros": {
            "random_forest": RF_PARAMS, "xgboost": {**XGB_PARAMS, "scale_pos_weight": "n_benign/n_ataque (train)"},
            "mlp": MLP_PARAMS,
            "isolation_forest": {**IFOREST_PARAMS, "entrenado_con": "solo flujos Benign de entrenamiento",
                                 "umbral": f"percentil {IFOREST_BENIGN_PERCENTILE} de la puntuacion en Benign de entrenamiento"},
            "lstm_gru": {**RNN_PARAMS, "validation_split": 0.2, "class_weight": "balanced", "umbral": 0.5},
        },
        "entorno": {"python": platform.python_version(), "maquina": platform.platform(), "procesador": platform.processor()},
        "seleccion_11_variables": selection,
        "resultados": results,
    }
    (out_dir / f"resultados_{tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    table = write_tables(results, out_dir, tag, len(y_eval))

    print(f"\n--- RANKING ({len(y_eval)} instancias de prueba comunes, {args.seeds} semillas) ---")
    for _, r in table.iterrows():
        res = results[r["clave"]]["resumen"]
        print(f"{r['rank']:2d}. {r['clave']:11s} acc {fmt(res['accuracy'])} | prec {fmt(res['precision'])} | "
              f"rec {fmt(res['recall'])} | f1 {fmt(res['f1'])} | auc {fmt(res['auc_roc'])}")
    if "rf11_multi" in results:
        print("\n--- Reporte multiclase del modelo desplegado (semilla 42) ---")
        print(results["rf11_multi"]["reporte_multiclase"])
    print(f"[OK] Resultados en {out_dir}: resultados_{tag}.json, tabla_{tag}.csv, tabla_{tag}.tex")


if __name__ == "__main__":
    main()
