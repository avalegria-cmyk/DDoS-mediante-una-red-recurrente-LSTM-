from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = ROOT.parent / "archive"
DEFAULT_OUTPUT_DIR = ROOT / "modelo_rf_compacto"

# Equivalentes directos a las 9 variables del detector ICMP:
# duracion, cantidad de paquetes/bytes, tasa de paquetes/bytes e IAT.
FEATURE_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un Random Forest compacto para DDoS usando pocas variables tipo ICMP."
    )
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows-per-file", type=int, default=15000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "source-holdout"],
        default="random",
        help=(
            "random imita el train_test_split 70/30 del detector ICMP; "
            "source-holdout entrena con *-training y evalua con *-testing."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary",
        help="binary detecta Benign/Ataque; multiclass detecta el tipo de ataque.",
    )
    return parser.parse_args()


def load_split(paths: list[Path], max_rows_per_file: int, random_state: int) -> pd.DataFrame:
    frames = []
    columns = [*FEATURE_COLUMNS, "Label"]
    for path in paths:
        df = pd.read_parquet(path, columns=columns)
        if len(df) > max_rows_per_file:
            df = df.sample(n=max_rows_per_file, random_state=random_state)
        df["source_file"] = path.name
        frames.append(df)
        print(f"{path.name}: {len(df)} filas")
    if not frames:
        raise FileNotFoundError("No se encontraron archivos Parquet para entrenar/evaluar.")
    return pd.concat(frames, ignore_index=True)


def normalize_label(value: object) -> str:
    label = str(value).strip()
    if label == "Benign":
        return "Benign"
    label = label.replace("DrDoS_", "")
    aliases = {
        "UDP-lag": "UDPLag",
        "UDP_Lag": "UDPLag",
        "LDAP": "LDAP",
        "MSSQL": "MSSQL",
        "NetBIOS": "NetBIOS",
        "UDP": "UDP",
        "Syn": "Syn",
    }
    return aliases.get(label, label)


def prepare_xy(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if task == "multiclass":
        y = df["Label"].map(normalize_label)
    else:
        y = df["Label"].astype(str).ne("Benign").astype(int)
    return X, y


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    all_paths = sorted(archive_dir.glob("*.parquet"))
    training_paths = sorted(archive_dir.glob("*-training.parquet"))
    testing_paths = sorted(archive_dir.glob("*-testing.parquet"))
    if not all_paths:
        raise FileNotFoundError(f"No hay archivos .parquet en {archive_dir}")

    print("--- CARGANDO DATASET CIC-DDoS2019 ---")
    print(f"Origen: {archive_dir}")
    print(f"Estrategia de split: {args.split_strategy}")

    if args.split_strategy == "random":
        full_df = load_split(all_paths, args.max_rows_per_file, args.random_state)
        if args.task == "multiclass":
            y_full = full_df["Label"].map(normalize_label)
        else:
            y_full = full_df["Label"].astype(str).ne("Benign").astype(int)
        train_df, test_df = train_test_split(
            full_df,
            test_size=0.30,
            random_state=args.random_state,
            stratify=y_full,
        )
    else:
        if not training_paths:
            raise FileNotFoundError(f"No hay *-training.parquet en {archive_dir}")
        if not testing_paths:
            raise FileNotFoundError(f"No hay *-testing.parquet en {archive_dir}")
        train_df = load_split(training_paths, args.max_rows_per_file, args.random_state)
        test_df = load_split(testing_paths, args.max_rows_per_file, args.random_state)

    X_train, y_train = prepare_xy(train_df, args.task)
    X_test, y_test = prepare_xy(test_df, args.task)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    print("\n--- ENTRENANDO RANDOM FOREST COMPACTO ---")
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.random_state,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    labels = sorted(y_test.unique().tolist(), key=lambda item: (item != "Benign", str(item)))
    if args.task == "binary":
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Ataque"],
            digits=4,
        )
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        matrix_summary = f"Matriz TN FP FN TP: {tn} {fp} {fn} {tp}"
        matrix_payload = [int(tn), int(fp), int(fn), int(tp)]
    else:
        report = classification_report(y_test, y_pred, labels=labels, digits=4)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        matrix_summary = f"Matriz de confusion multiclase: {cm.shape[0]}x{cm.shape[1]}"
        matrix_payload = cm.astype(int).tolist()

    print("\n" + report)
    print(matrix_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "xgboost.pkl")
    joblib.dump({"imputer": imputer, "scaler": scaler}, output_dir / "preprocessors.pkl")
    (output_dir / "feature_cols.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    class_labels = [str(item) for item in model.classes_.tolist()]
    (output_dir / "class_labels.json").write_text(
        json.dumps(class_labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "model_kind": f"random_forest_compacto_{args.task}",
        "compat_model_kind_for_app": "xgboost",
        "archive_dir": str(archive_dir),
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "class_labels": class_labels,
        "split_strategy": args.split_strategy,
        "task": args.task,
        "label_rule": (
            "Label normalizada por tipo de ataque"
            if args.task == "multiclass"
            else "Label != 'Benign' => Ataque(1), Label == 'Benign' => Benign(0)"
        ),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_class_counts": y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "confusion_matrix": matrix_payload,
        "confusion_matrix_labels": labels if args.task == "multiclass" else ["Benign", "Ataque"],
        "classification_report": report,
        "note": "Los nombres xgboost.pkl/feature_cols.json se usan solo por compatibilidad con app.py.",
    }
    (output_dir / "info_rf_compacto.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "training_report.txt").write_text(
        "\n".join(
            [
                "REPORTE RANDOM FOREST COMPACTO DDoS",
                f"Dataset: {archive_dir}",
                f"Filas train: {len(train_df)}",
                f"Filas test: {len(test_df)}",
                f"Features: {len(FEATURE_COLUMNS)}",
                matrix_summary,
                "",
                report,
            ]
        ),
        encoding="utf-8",
    )

    print(f"\n[OK] Modelo compacto guardado en: {output_dir}")
    print("Para usarlo en la app:")
    print(f"MODEL_KIND=xgboost MODEL_DIR='{output_dir}' python3 -m uvicorn app:app --reload")


if __name__ == "__main__":
    main()
