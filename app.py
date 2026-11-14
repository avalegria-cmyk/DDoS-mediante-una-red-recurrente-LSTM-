from __future__ import annotations

import ipaddress
import json
import os
import platform
import shlex
import gzip
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "modelo_rf_multiclase"))
FLOW_DIR = Path(os.getenv("FLOW_DIR", BASE_DIR / "data" / "flows"))
STATE_DIR = Path(os.getenv("STATE_DIR", BASE_DIR / "data" / "state"))
WEB_DIR = Path(os.getenv("WEB_DIR", BASE_DIR / "web"))

MODEL_KIND = os.getenv("MODEL_KIND", "xgboost").lower()
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost.pkl"
XGBOOST_MODEL_GZ_PATH = MODEL_DIR / "xgboost.pkl.gz"
PREPROCESSORS_PATH = MODEL_DIR / "preprocessors.pkl"
XGBOOST_FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.json"
MODEL_INFO_PATH = MODEL_DIR / "info_rf_compacto.json"
LSTM_MODEL_PATH = MODEL_DIR / os.getenv("LSTM_MODEL_FILE", "modelo_lstm_ddos.keras")
GRU_MODEL_PATH = MODEL_DIR / "modelo_gru_ddos.keras"
LSTM_SCALER_PATH = MODEL_DIR / "scaler_lstm_gru.pkl"
LSTM_FEATURE_COLS_PATH = MODEL_DIR / "feature_cols_lstm_gru.json"
LSTM_METADATA_PATH = MODEL_DIR / "lstm_runtime_metadata.json"

PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.65"))
AUTO_BLOCK = os.getenv("AUTO_BLOCK", "false").lower() == "true"
_INSTALLED_BLOCK_SCRIPT = Path("/usr/local/sbin/ddos-block-ip")
BLOCK_SCRIPT = os.getenv(
    "BLOCK_SCRIPT",
    str(_INSTALLED_BLOCK_SCRIPT if _INSTALLED_BLOCK_SCRIPT.exists() else BASE_DIR / "scripts" / "ddos-block-ip"),
)
# Los bloqueos son temporales: caducan a los BLOCK_TTL_SECONDS (0 = sin caducidad).
BLOCK_TTL_SECONDS = int(os.getenv("BLOCK_TTL_SECONDS", "300"))
BLOCK_USE_SUDO = os.getenv("BLOCK_USE_SUDO", "true").lower() == "true"
MITIGATION_LOG_PATH = STATE_DIR / "eventos_mitigacion.jsonl"
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "RF-11" if MODEL_KIND == "xgboost" else MODEL_KIND.upper())
SCAN_INTERVAL_SECONDS = float(os.getenv("SCAN_INTERVAL_SECONDS", "2"))
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "300"))
DEFAULT_WHITELIST_IPS = ",".join(
    [
        "127.0.0.1",
        "17.248.201.71",   # Apple
        "57.144.199.33",   # Meta/Facebook
        "104.18.32.47",    # Cloudflare
        "140.82.114.26",   # GitHub
        "172.64.148.235",  # Cloudflare
    ]
)
WHITELIST_IPS = {ip.strip() for ip in os.getenv("WHITELIST_IPS", DEFAULT_WHITELIST_IPS).split(",") if ip.strip()}


def _local_addresses() -> set[str]:
    """Direcciones propias del equipo (se usan si PROTECTED_IPS no se define)."""
    found: set[str] = set()
    try:
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.0.2.1", 9))  # no envia nada; solo elige la interfaz de salida
        found.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return found


# Direcciones del servidor protegido: nunca se bloquean. Si un flujo aparece con una de ellas como
# origen (ocurre cuando la ventana de captura empieza a mitad de una conexion y el primer paquete
# visto es la respuesta del servidor), el flujo se atribuye al otro extremo.
PROTECTED_IPS = {ip.strip() for ip in os.getenv("PROTECTED_IPS", "").split(",") if ip.strip()} or _local_addresses()
IGNORE_LOCAL_NOISE = os.getenv("IGNORE_LOCAL_NOISE", "true").lower() == "true"
LOCAL_NOISE_PORTS = {
    int(port.strip())
    for port in os.getenv("LOCAL_NOISE_PORTS", "137,138,5353,1900,3289").split(",")
    if port.strip().isdigit()
}
RATE_ATTACK_PPS = float(os.getenv("RATE_ATTACK_PPS", "1000"))
RATE_ATTACK_ANY_PORT = os.getenv("RATE_ATTACK_ANY_PORT", "true").lower() == "true"
MODEL_ATTACK_MIN_PPS = float(os.getenv("MODEL_ATTACK_MIN_PPS", "10"))
PROTECTED_PORTS = {
    int(port.strip())
    for port in os.getenv("PROTECTED_PORTS", "80,443,8080").split(",")
    if port.strip().isdigit()
}
DEFAULT_CAPTURE_INTERFACE = "auto" if platform.system() == "Windows" else "en0"
CAPTURE_INTERFACE = os.getenv("CAPTURE_INTERFACE", DEFAULT_CAPTURE_INTERFACE)
CAPTURE_SECONDS = int(os.getenv("CAPTURE_SECONDS", "15"))
# Regla por tasa: se evalua por IP de origen sobre la ventana de captura, no por flujo.
# Un flujo corto y legitimo (p. ej. una peticion HTTP de 1 ms) tiene miles de paquetes/s
# "por flujo", pero su origen envia muy pocos paquetes y abre pocas conexiones por ventana.
# Limite de paquetes por ventana: bajo una inundacion la ventana se cierra antes (reaccion mas
# rapida) y la conversion a flujos queda acotada. 0 = sin limite.
CAPTURE_MAX_PACKETS = int(os.getenv("CAPTURE_MAX_PACKETS", "50000"))
CAPTURE_SNAPLEN = int(os.getenv("CAPTURE_SNAPLEN", "0"))  # 0 = paquete completo (igual que en el entrenamiento)
RATE_WINDOW_SECONDS = float(os.getenv("RATE_WINDOW_SECONDS", str(CAPTURE_SECONDS)))
RATE_ATTACK_FLOWS_PER_SECOND = float(os.getenv("RATE_ATTACK_FLOWS_PER_SECOND", "20"))
CAPTURE_USE_SUDO = os.getenv("CAPTURE_USE_SUDO", "true").lower() == "true"
PCAP_DIR = Path(os.getenv("PCAP_DIR", BASE_DIR / "data" / "pcap"))
CICFLOWMETER_CMD = os.getenv("CICFLOWMETER_CMD", "")
CAPTURE_COMMAND = os.getenv("CAPTURE_COMMAND", "")
PROCESS_EXISTING_FILES = os.getenv("PROCESS_EXISTING_FILES", "false").lower() == "true"
KEEP_CAPTURE_FILES = os.getenv("KEEP_CAPTURE_FILES", "false").lower() == "true"

IP_COLUMNS = ["Src IP", "Source IP", "src_ip", "SourceIP", "Source", "source_ip"]
DST_COLUMNS = ["Dst IP", "Destination IP", "dst_ip", "DestinationIP", "Destination", "dest_ip"]
PORT_COLUMNS = ["Dst Port", "Destination Port", "dst_port", "DestinationPort", "port"]
TIME_COLUMNS = ["Timestamp", "Flow Start", "timestamp", "time"]

# Columnas producidas por el paquete Python cicflowmeter 0.5.0.
PY_CICFLOWMETER_COLUMNS = {
    "protocol": "Protocol",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Fwd Packets Length Total",
    "totlen_bwd_pkts": "Bwd Packets Length Total",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Packet Length Min",
    "pkt_len_max": "Packet Length Max",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwr_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Avg Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init Fwd Win Bytes",
    "init_bwd_win_byts": "Init Bwd Win Bytes",
    "fwd_act_data_pkts": "Fwd Act Data Packets",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
    "src_ip": "Src IP",
    "dst_ip": "Dst IP",
    "dst_port": "Dst Port",
    "timestamp": "Timestamp",
}

PY_CICFLOWMETER_TIME_COLUMNS = {
    "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
}


class BlockRequest(BaseModel):
    ip: str
    reason: str = "Bloqueo manual desde dashboard"


class DetectorState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.traffic: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self.blocked_ips: list[dict[str, Any]] = []
        self.processed_files: set[str] = set()
        self.last_error: str | None = None
        self.rows_processed = 0
        self.files_processed = 0
        self.started_at = datetime.now()
        self.flow_windows: dict[str, deque[np.ndarray]] = {}
        self.capture_enabled = os.getenv("CAPTURE_AUTO_START", "false").lower() == "true"
        self.capture_running = False
        self.capture_last_file: str | None = None
        self.capture_last_error: str | None = None
        self.capture_cycles = 0
        self.capture_failed_cycles = 0
        self.capture_windows: dict[str, float] = {}  # duracion real de la ventana de cada CSV

    def add_event(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.traffic.appendleft(event)
            self.rows_processed += 1

    def add_block(self, ip: str, reason: str, risk: int, source: str) -> bool:
        """Registra la IP. Devuelve False si ya estaba registrada."""
        with self.lock:
            if any(item["ip"] == ip for item in self.blocked_ips):
                return False
            now = time.time()
            item = {
                "ip": ip,
                "reason": reason,
                "risk": risk,
                "since": now_time(),
                "sinceEpoch": now,
                "expiresEpoch": now + BLOCK_TTL_SECONDS if BLOCK_TTL_SECONDS > 0 else None,
                "status": "Detectada",
                "source": source,
            }
            self.blocked_ips.insert(0, item)
            return True

    def set_block_status(self, ip: str, status: str) -> None:
        with self.lock:
            for item in self.blocked_ips:
                if item["ip"] == ip:
                    item["status"] = status

    def pop_expired_blocks(self) -> list[dict[str, Any]]:
        now = time.time()
        with self.lock:
            expired = [i for i in self.blocked_ips if i.get("expiresEpoch") and i["expiresEpoch"] <= now]
            self.blocked_ips = [i for i in self.blocked_ips if i not in expired]
            return expired

    def remove_block(self, ip: str) -> bool:
        with self.lock:
            initial_count = len(self.blocked_ips)
            self.blocked_ips = [item for item in self.blocked_ips if item["ip"] != ip]
            return len(self.blocked_ips) != initial_count


state = DetectorState()

if MODEL_KIND in {"lstm", "gru"}:
    from tensorflow.keras.models import load_model

    runtime_metadata = json.loads(LSTM_METADATA_PATH.read_text(encoding="utf-8"))
    feature_cols = json.loads(LSTM_FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    timesteps = int(runtime_metadata.get("timesteps", 10))
    sequence_model_path = GRU_MODEL_PATH if MODEL_KIND == "gru" else LSTM_MODEL_PATH
    model = load_model(sequence_model_path)
    scaler_lstm = joblib.load(LSTM_SCALER_PATH)
    preprocessors = None
else:
    timesteps = 1
    if not XGBOOST_MODEL_PATH.exists() and XGBOOST_MODEL_GZ_PATH.exists():
        with gzip.open(XGBOOST_MODEL_GZ_PATH, "rb") as source, XGBOOST_MODEL_PATH.open("wb") as target:
            shutil.copyfileobj(source, target)
    model = joblib.load(XGBOOST_MODEL_PATH)
    preprocessors = joblib.load(PREPROCESSORS_PATH)
    feature_cols = json.loads(XGBOOST_FEATURE_COLS_PATH.read_text(encoding="utf-8"))
    scaler_lstm = None

model_metadata = {}
if MODEL_INFO_PATH.exists():
    model_metadata = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))

class_labels: list[str] = []
if CLASS_LABELS_PATH.exists():
    class_labels = json.loads(CLASS_LABELS_PATH.read_text(encoding="utf-8"))
elif hasattr(model, "classes_"):
    class_labels = [str(item) for item in model.classes_.tolist()]

app = FastAPI(title="DDoS Shield Real Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")


@app.on_event("startup")
def startup() -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PCAP_DIR.mkdir(parents=True, exist_ok=True)
    if not PROCESS_EXISTING_FILES:
        state.processed_files.update(str(path) for path in FLOW_DIR.glob("*.csv"))
    thread = threading.Thread(target=watch_flow_directory, daemon=True)
    thread.start()
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    expiry_thread = threading.Thread(target=block_expiry_loop, daemon=True)
    expiry_thread.start()


@app.get("/")
def dashboard() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard no instalado")
    return FileResponse(index)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": f"real-flow-detector-{MODEL_KIND}",
        "model": str(sequence_model_path if MODEL_KIND in {"lstm", "gru"} else XGBOOST_MODEL_PATH),
        "flow_dir": str(FLOW_DIR),
        "auto_block": AUTO_BLOCK,
        "block_ttl_seconds": BLOCK_TTL_SECONDS,
        "block_script": BLOCK_SCRIPT,
        "threshold": PREDICTION_THRESHOLD,
        "protected_ips": sorted(PROTECTED_IPS),
        "rate_rule": {"window_s": RATE_WINDOW_SECONDS, "source_pps": RATE_ATTACK_PPS,
                      "source_flows_per_s": RATE_ATTACK_FLOWS_PER_SECOND},
        "timesteps": timesteps,
        "feature_count": len(feature_cols),
        "capture_enabled": state.capture_enabled,
        "capture_interface": CAPTURE_INTERFACE,
        "platform": platform.system(),
        "cicflowmeter_configured": bool(CICFLOWMETER_CMD),
        "last_error": state.last_error,
        "whitelist_ips": sorted(WHITELIST_IPS),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    with state.lock:
        traffic = list(state.traffic)
        blocked = list(state.blocked_ips)
        rows_processed = state.rows_processed
        files_processed = state.files_processed

    attacks = [item for item in traffic if item["decision"] in {"Bloquear", "Monitorear"}]
    avg_risk = round(sum(item["risk"] for item in traffic) / max(len(traffic), 1))
    total_pps = sum(item.get("pps", 0) for item in traffic[:30])
    return {
        "server": platform.node(),
        "status": "Ataque activo" if any(item["decision"] == "Bloquear" for item in traffic[:20]) else "Operativo",
        "totalPps": total_pps,
        "avgRisk": avg_risk,
        "attacksNow": len(attacks),
        "blockedCount": len(blocked),
        "protectedPorts": 0,
        "rowsProcessed": rows_processed,
        "filesProcessed": files_processed,
    }


@app.get("/api/traffic")
def traffic() -> dict[str, Any]:
    with state.lock:
        return {"traffic": list(state.traffic)}


@app.get("/api/blocked-ips")
def blocked_ips() -> dict[str, Any]:
    with state.lock:
        return {"blockedIps": list(state.blocked_ips)}


@app.post("/api/blocked-ips")
def block_ip(request: BlockRequest) -> dict[str, Any]:
    validate_ip(request.ip)
    apply_block(request.ip, request.reason, 100, "manual")
    if AUTO_BLOCK and any(i["ip"] == request.ip and i["status"] != "Bloqueada" for i in state.blocked_ips):
        state.remove_block(request.ip)
        raise HTTPException(status_code=500, detail=state.last_error or "No se pudo aplicar el bloqueo")
    return {"ok": True, "ip": request.ip}


@app.delete("/api/blocked-ips/{ip}")
def unblock_ip(ip: str) -> dict[str, Any]:
    validate_ip(ip)
    removed = state.remove_block(ip)
    firewall_ok = None
    if removed and AUTO_BLOCK:
        firewall_ok = run_block_script(ip, "unblock")
    if removed:
        log_mitigation({"evento": "desbloqueo_manual", "ip": ip, "firewall_ok": firewall_ok})
    return {"ok": True, "ip": ip, "removed": removed, "firewallOk": firewall_ok}


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    if MODEL_KIND in {"lstm", "gru"}:
        pipeline = [
            f"Capturar trafico real del servidor {platform.system()}",
            "Convertir paquetes a flujos con CICFlowMeter o extractor compatible",
            "Generar CSV con las mismas columnas del entrenamiento",
            "Aplicar el scaler_lstm_gru.pkl exportado desde el entrenamiento LSTM",
            f"Acumular {timesteps} flujos por IP para crear un tensor (1, {timesteps}, {len(feature_cols)})",
            f"Enviar la ventana temporal al modelo {MODEL_KIND.upper()} entrenado",
            "Mostrar IP atacante y aplicar bloqueo si AUTO_BLOCK=true",
        ]
    else:
        pipeline = [
            f"Capturar trafico real del servidor {platform.system()}",
            "Convertir paquetes a flujos con CICFlowMeter o extractor compatible",
            "Normalizar nombres de columnas snake_case a formato CIC-DDoS2019",
            f"Aplicar imputador y scaler sobre {len(feature_cols)} caracteristicas",
            "Enviar cada flujo al Random Forest multiclase",
            "Mostrar clase predicha y bloquear si la probabilidad de ataque supera el umbral",
        ]

    return {
        "metadata": {
            "xgboost": {
                "available": XGBOOST_MODEL_PATH.exists(),
                "path": str(XGBOOST_MODEL_PATH),
                "featureCount": len(feature_cols),
                "modelType": type(model).__name__,
                "classLabels": class_labels,
                "modelMetadata": model_metadata,
            },
            "lstm": {
                "available": LSTM_MODEL_PATH.exists() and LSTM_SCALER_PATH.exists(),
                "path": str(LSTM_MODEL_PATH),
                "scaler": str(LSTM_SCALER_PATH),
                "timesteps": timesteps,
                "inputShape": [1, timesteps, len(feature_cols)],
            },
            "gru": {
                "available": GRU_MODEL_PATH.exists() and LSTM_SCALER_PATH.exists(),
                "path": str(GRU_MODEL_PATH),
                "scaler": str(LSTM_SCALER_PATH),
                "timesteps": timesteps,
                "inputShape": [1, timesteps, len(feature_cols)],
            },
            "preprocessors": {
                "available": PREPROCESSORS_PATH.exists(),
                "path": str(PREPROCESSORS_PATH),
            },
            "activeModel": MODEL_KIND,
        },
        "pipeline": pipeline,
        "requiredFeatures": feature_cols,
    }


@app.get("/api/capture/status")
def capture_status() -> dict[str, Any]:
    with state.lock:
        return {
            "failedCycles": state.capture_failed_cycles,
        "maxPacketsPerWindow": CAPTURE_MAX_PACKETS,
        "enabled": state.capture_enabled,
            "running": state.capture_running,
            "interface": CAPTURE_INTERFACE,
            "captureSeconds": CAPTURE_SECONDS,
            "pcapDir": str(PCAP_DIR),
            "flowDir": str(FLOW_DIR),
            "cicflowmeterConfigured": bool(CICFLOWMETER_CMD),
            "cicflowmeterCommand": CICFLOWMETER_CMD,
            "lastFile": state.capture_last_file,
            "lastError": state.capture_last_error,
            "cycles": state.capture_cycles,
        }


@app.post("/api/capture/start")
def start_capture() -> dict[str, Any]:
    with state.lock:
        state.capture_enabled = True
        state.capture_last_error = None
    return {"ok": True, "enabled": True}


@app.post("/api/capture/stop")
def stop_capture() -> dict[str, Any]:
    with state.lock:
        state.capture_enabled = False
    return {"ok": True, "enabled": False}


def watch_flow_directory() -> None:
    while True:
        try:
            for csv_path in sorted(FLOW_DIR.glob("*.csv")):
                if str(csv_path) in state.processed_files:
                    continue
                if csv_path.stat().st_size <= 0:
                    state.processed_files.add(str(csv_path))
                    cleanup_capture_files(csv_path)
                    continue
                process_flow_file(csv_path)
                state.processed_files.add(str(csv_path))
                state.files_processed += 1
                cleanup_capture_files(csv_path)
        except Exception as exc:
            state.last_error = str(exc)
        time.sleep(SCAN_INTERVAL_SECONDS)


def capture_loop() -> None:
    while True:
        if not state.capture_enabled:
            time.sleep(1)
            continue
        if state.capture_running:
            time.sleep(1)
            continue
        try:
            run_capture_cycle()
        except Exception as exc:
            # Un ciclo fallido (p. ej. una conversion que excede el tiempo durante una inundacion)
            # no debe apagar la captura: se registra el error y se continua con la siguiente ventana.
            with state.lock:
                state.capture_last_error = str(exc)
                state.capture_running = False
                state.capture_failed_cycles += 1
            time.sleep(3)


def run_capture_cycle() -> None:
    with state.lock:
        state.capture_running = True
        state.capture_last_error = None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    pcap_path = PCAP_DIR / f"captura-{timestamp}.pcap"
    csv_path = FLOW_DIR / f"flows-{timestamp}.csv"
    partial_csv_path = csv_path.with_suffix(".partial")

    capture_started = time.time()
    try:
        if CAPTURE_COMMAND:
            capture_command = CAPTURE_COMMAND.format(
                interface=shell_quote(CAPTURE_INTERFACE),
                duration=CAPTURE_SECONDS,
                pcap=shell_quote(str(pcap_path)),
            )
            subprocess.run(capture_command, shell=True, check=True, timeout=CAPTURE_SECONDS + 15)
        elif platform.system() == "Windows":
            raise RuntimeError("CAPTURE_COMMAND no esta configurado para Windows. Ejecuta iniciar.bat.")
        else:
            capture_cmd = [
                "/usr/sbin/tcpdump",
                "-i",
                CAPTURE_INTERFACE,
                "-nn",
                *(["-s", str(CAPTURE_SNAPLEN)] if CAPTURE_SNAPLEN > 0 else []),
                *(["-c", str(CAPTURE_MAX_PACKETS)] if CAPTURE_MAX_PACKETS > 0 else []),
                "-G",
                str(CAPTURE_SECONDS),
                "-W",
                "1",
                "-w",
                str(pcap_path),
                "not",
                "port",
                "4100",
            ]
            if CAPTURE_USE_SUDO and getattr(os, "geteuid", lambda: 1)() != 0:
                permission = subprocess.run(
                    ["sudo", "-n", "true"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if permission.returncode != 0:
                    raise RuntimeError(
                        "macOS requiere autorizacion de administrador. "
                        "Ejecuta macos/iniciar_detector_admin.sh con sudo."
                    )
                capture_cmd = ["sudo", "-n", *capture_cmd]
            try:
                subprocess.run(capture_cmd, check=True, timeout=CAPTURE_SECONDS + 8)
            except subprocess.TimeoutExpired:
                if not pcap_path.exists() or pcap_path.stat().st_size <= 24:
                    pcap_path.unlink(missing_ok=True)
                    with state.lock:
                        state.capture_last_error = None
                    return

        capture_elapsed = max(time.time() - capture_started, 0.5)
        with state.lock:
            state.capture_windows[csv_path.name] = capture_elapsed
            while len(state.capture_windows) > 200:
                state.capture_windows.pop(next(iter(state.capture_windows)))

        if not CICFLOWMETER_CMD:
            raise RuntimeError(
                "CICFLOWMETER_CMD no esta configurado. Instala CICFlowMeter y define el comando para convertir PCAP a CSV."
            )

        command = CICFLOWMETER_CMD.format(
            pcap=shell_quote(str(pcap_path)),
            csv=shell_quote(str(partial_csv_path)),
            flow_dir=shell_quote(str(FLOW_DIR)),
        )
        subprocess.run(command, shell=True, check=True, timeout=120)
        if not partial_csv_path.exists():
            raise RuntimeError("CICFlowMeter termino sin generar el CSV esperado.")
        partial_csv_path.replace(csv_path)

        with state.lock:
            state.capture_last_file = str(csv_path)
            state.capture_cycles += 1
    finally:
        with state.lock:
            state.capture_running = False


def process_flow_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if df.empty:
        return

    df = normalize_cicflowmeter_columns(df)

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} no tiene el formato del entrenamiento. "
            f"Faltan columnas: {missing[:8]}"
        )

    if MODEL_KIND in {"lstm", "gru"}:
        process_sequence_file(csv_path, df)
    else:
        process_xgboost_file(csv_path, df)


def cleanup_capture_files(csv_path: Path) -> None:
    if KEEP_CAPTURE_FILES:
        return

    csv_path.unlink(missing_ok=True)
    if csv_path.name.startswith("flows-"):
        pcap_name = "captura-" + csv_path.name.removeprefix("flows-").removesuffix(".csv") + ".pcap"
        (PCAP_DIR / pcap_name).unlink(missing_ok=True)


def process_xgboost_file(csv_path: Path, df: pd.DataFrame) -> None:
    records = df.to_dict(orient="records")
    source_rates = source_rate_stats(df, state.capture_windows.get(csv_path.name))
    X = prepare_xgboost_features(df)
    probability_matrix = model.predict_proba(X)
    multiclass_labels = class_labels if "Benign" in class_labels and len(class_labels) > 2 else []

    for row, probabilities in zip(records, probability_matrix):
        probabilities = np.asarray(probabilities, dtype=float)
        class_probability_map = {
            str(label): round(float(prob), 6)
            for label, prob in zip(class_labels, probabilities)
        }

        if multiclass_labels:
            predicted_index = int(np.argmax(probabilities))
            predicted_label = str(multiclass_labels[predicted_index])
            attack_candidates = [
                (str(label), float(prob))
                for label, prob in zip(multiclass_labels, probabilities)
                if str(label) != "Benign"
            ]
            top_attack_label, top_attack_class_probability = max(
                attack_candidates,
                key=lambda item: item[1],
                default=("DDoS/DoS", 0.0),
            )
            attack_probability = float(
                sum(prob for _, prob in attack_candidates)
            )
            probability = attack_probability
            attack = predicted_label if predicted_label != "Benign" else top_attack_label
        else:
            probability = float(probabilities[1])
            predicted_label = "DDoS/DoS" if probability >= PREDICTION_THRESHOLD else "Benign"
            attack = predicted_label
            top_attack_class_probability = probability

        src_ip = pick_value(row, IP_COLUMNS, "desconocida")
        dst_ip = pick_value(row, DST_COLUMNS, "servidor")
        port = pick_value(row, PORT_COLUMNS, 0)
        event_time = pick_value(row, TIME_COLUMNS, now_time())
        pps = safe_number(row.get("Flow Packets/s", 0))
        bps = safe_number(row.get("Flow Bytes/s", 0))

        reversed_flow = str(src_ip) in PROTECTED_IPS and str(dst_ip) not in PROTECTED_IPS
        src_ip = remote_endpoint(str(src_ip), str(dst_ip))
        if reversed_flow:
            # Las variables de un flujo visto al reves tienen los sentidos intercambiados: no se usa
            # la probabilidad del modelo para decidir; el flujo solo cuenta para la tasa del extremo remoto.
            probability = 0.0
        source_rate = source_rates.get(str(src_ip), {"pps": 0.0, "fps": 0.0})
        rate_attack = is_rate_attack_source(source_rate)
        local_noise = is_local_noise_flow(row, dst_ip, port, pps, bps)
        low_rate_model_noise = pps < MODEL_ATTACK_MIN_PPS and not rate_attack
        risk = round(probability * 100)
        if rate_attack:
            risk = max(risk, rate_risk_score(source_rate))

        decision = "Bloquear" if (probability >= PREDICTION_THRESHOLD and not low_rate_model_noise) or rate_attack else "Permitir"
        if local_noise:
            probability = 0.0
            risk = 0
            attack = "Ruido local ignorado"
            decision = "Permitir"
        elif low_rate_model_noise:
            attack = "Benign"
            decision = "Permitir"
            risk = min(risk, 30)
        elif rate_attack and probability < PREDICTION_THRESHOLD:
            attack = "DDoS/DoS por tasa"
        elif decision == "Permitir":
            attack = "Benign"

        event = {
            "id": f"{csv_path.name}-{len(state.traffic)}-{time.time_ns()}",
            "time": str(event_time),
            "sourceIp": str(src_ip),
            "destIp": str(dst_ip),
            "protocol": str(row.get("Protocol", "TCP")),
            "attack": attack,
            "pps": round(pps),
            "bps": round(bps),
            "sourcePps": round(source_rate["pps"], 2),
            "sourceFlowsPerSecond": round(source_rate["fps"], 2),
            "port": int(safe_number(port)),
            "risk": risk,
            "decision": decision,
            "probability": round(probability, 6),
            "sourceFile": csv_path.name,
            "modelKind": MODEL_KIND,
            "predictedClass": predicted_label,
            "topAttackClass": attack if attack != "DDoS/DoS por tasa" else "rate-rule",
            "topAttackClassProbability": round(float(top_attack_class_probability), 6),
            "classProbabilities": class_probability_map,
        }
        state.add_event(event)

        if decision == "Bloquear" and src_ip != "desconocida" and not is_whitelisted(str(src_ip)):
            reason = f"Modelo {MODEL_DISPLAY_NAME} detecto {attack} con riesgo {risk}%"
            apply_block(str(src_ip), reason, risk, "model", flow_time=str(event_time), source_file=csv_path.name,
                        attack=str(attack), probability=float(probability), pps=float(pps),
                        source_pps=float(source_rate["pps"]), source_flows_per_s=float(source_rate["fps"]),
                        disparo="regla_por_tasa" if rate_attack and probability < PREDICTION_THRESHOLD else "modelo")


def process_sequence_file(csv_path: Path, df: pd.DataFrame) -> None:
    source_rates = source_rate_stats(df, state.capture_windows.get(csv_path.name))
    records = df.to_dict(orient="records")
    X_scaled = prepare_lstm_features(df)
    pending_events: list[dict[str, Any]] = []
    pending_sequences: list[np.ndarray] = []

    for row, vector in zip(records, X_scaled):
        src_ip = str(pick_value(row, IP_COLUMNS, "desconocida"))
        dst_ip = pick_value(row, DST_COLUMNS, "servidor")
        port = pick_value(row, PORT_COLUMNS, 0)
        event_time = pick_value(row, TIME_COLUMNS, now_time())
        pps = safe_number(row.get("Flow Packets/s", 0))
        bps = safe_number(row.get("Flow Bytes/s", 0))

        if is_local_noise_flow(row, dst_ip, port, pps, bps):
            event = {
                "id": f"{csv_path.name}-{len(state.traffic)}-{time.time_ns()}",
                "time": str(event_time),
                "sourceIp": src_ip,
                "destIp": str(dst_ip),
                "protocol": str(row.get("Protocol", "TCP")),
                "attack": "Ruido local ignorado",
                "pps": round(pps),
                "bps": round(bps),
                "port": int(safe_number(port)),
                "risk": 0,
                "decision": "Permitir",
                "probability": 0.0,
                "sourceFile": csv_path.name,
                "modelKind": MODEL_KIND,
                "windowSize": 0,
                "ignored": True,
            }
            state.add_event(event)
            continue

        window = state.flow_windows.setdefault(src_ip, deque(maxlen=timesteps))
        window.append(np.asarray(vector, dtype=np.float32))

        if len(window) < timesteps:
            event = {
                "id": f"{csv_path.name}-{len(state.traffic)}-{time.time_ns()}",
                "time": str(event_time),
                "sourceIp": src_ip,
                "destIp": str(dst_ip),
                "protocol": str(row.get("Protocol", "TCP")),
                "attack": f"Recolectando ventana {len(window)}/{timesteps}",
                "pps": round(pps),
                "bps": round(bps),
                "port": int(safe_number(port)),
                "risk": 0,
                "decision": "Permitir",
                "probability": 0.0,
                "sourceFile": csv_path.name,
                "modelKind": MODEL_KIND,
                "windowSize": len(window),
            }
            state.add_event(event)
            continue
        else:
            sequence = np.asarray(window, dtype=np.float32).reshape(1, timesteps, len(feature_cols))
            pending_sequences.append(sequence[0])

        event = {
            "id": f"{csv_path.name}-{len(state.traffic)}-{time.time_ns()}",
            "time": str(event_time),
            "sourceIp": src_ip,
            "destIp": str(dst_ip),
            "protocol": str(row.get("Protocol", "TCP")),
            "attack": "Pendiente",
            "pps": round(pps),
            "bps": round(bps),
            "port": int(safe_number(port)),
            "risk": 0,
            "decision": "Permitir",
            "probability": 0.0,
            "sourceFile": csv_path.name,
            "modelKind": MODEL_KIND,
            "windowSize": len(window),
        }
        pending_events.append(event)

    if pending_sequences:
        batch = np.asarray(pending_sequences, dtype=np.float32)
        probabilities = model.predict(batch, verbose=0, batch_size=256).ravel()
    else:
        probabilities = []

    for event, probability in zip(pending_events, probabilities):
        probability = float(probability)
        model_attack = probability >= PREDICTION_THRESHOLD
        source_rate = source_rates.get(str(event["sourceIp"]), {"pps": 0.0, "fps": 0.0})
        rate_attack = is_rate_attack_source(source_rate)
        risk = round(probability * 100)
        if rate_attack:
            risk = max(risk, rate_risk_score(source_rate))
        decision = "Bloquear" if model_attack or rate_attack else "Permitir"
        event["probability"] = round(probability, 6)
        event["risk"] = risk
        if model_attack:
            event["attack"] = "DDoS/DoS"
        elif rate_attack:
            event["attack"] = "DDoS/DoS por tasa"
        else:
            event["attack"] = "Benign"
        event["decision"] = decision
        state.add_event(event)

        src_ip = event["sourceIp"]
        if decision == "Bloquear" and src_ip != "desconocida" and not is_whitelisted(src_ip):
            reason = f"Detector {MODEL_DISPLAY_NAME} detecto ataque con riesgo {risk}%"
            apply_block(src_ip, reason, risk, MODEL_KIND, flow_time=str(event.get("time", "")),
                        source_file=csv_path.name, attack=str(event["attack"]))


def prepare_xgboost_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X_imp = pd.DataFrame(preprocessors["imputer"].transform(X), columns=feature_cols)
    X_scaled = pd.DataFrame(preprocessors["scaler"].transform(X_imp), columns=feature_cols)
    return X_scaled


def prepare_lstm_features(df: pd.DataFrame) -> np.ndarray:
    X = df[feature_cols].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    return scaler_lstm.transform(X).astype("float32")


def is_local_noise_flow(row: dict[str, Any], dst_ip: Any, port: Any, pps: float, bps: float) -> bool:
    if not IGNORE_LOCAL_NOISE:
        return False

    port_number = int(safe_number(port))
    dst_value = str(dst_ip)
    duration = safe_number(row.get("Flow Duration", row.get("flow_duration", 0)))
    total_packets = (
        safe_number(row.get("Total Fwd Packets", row.get("tot_fwd_pkts", 0)))
        + safe_number(row.get("Total Backward Packets", row.get("tot_bwd_pkts", 0)))
    )

    if is_non_unicast_ip(dst_value):
        return True
    if port_number in LOCAL_NOISE_PORTS and duration <= 0 and total_packets <= 2 and pps <= 0 and bps <= 0:
        return True
    return False


def source_rate_stats(df: pd.DataFrame, window_seconds: float | None = None) -> dict[str, dict[str, float]]:
    """Paquetes/s enviados y flujos nuevos/s por IP de origen en la ventana de captura.

    window_seconds es la duracion real de la ventana (puede ser menor que CAPTURE_SECONDS
    cuando se alcanza CAPTURE_MAX_PACKETS); si no se conoce se usa RATE_WINDOW_SECONDS.
    """
    src_col = next((col for col in IP_COLUMNS if col in df.columns), None)
    if src_col is None or df.empty:
        return {}
    work = df
    port_col = next((col for col in PORT_COLUMNS if col in df.columns), None)
    if not RATE_ATTACK_ANY_PORT and port_col is not None:
        work = df[pd.to_numeric(df[port_col], errors="coerce").isin(PROTECTED_PORTS)]
    if "Total Fwd Packets" in work.columns:
        sent = pd.to_numeric(work["Total Fwd Packets"], errors="coerce").fillna(0)
    else:
        sent = pd.Series(0.0, index=work.index)
    sources = work[src_col].astype(str)
    dst_col = next((col for col in DST_COLUMNS if col in work.columns), None)
    if dst_col is not None and PROTECTED_IPS:
        # Flujos vistos "al reves": se atribuyen al extremo remoto y se cuentan sus paquetes (los de retorno).
        reversed_flow = sources.isin(PROTECTED_IPS) & ~work[dst_col].astype(str).isin(PROTECTED_IPS)
        if reversed_flow.any() and "Total Backward Packets" in work.columns:
            back = pd.to_numeric(work["Total Backward Packets"], errors="coerce").fillna(0)
            sent = sent.where(~reversed_flow, back)
        sources = sources.where(~reversed_flow, work[dst_col].astype(str))
    grouped = pd.DataFrame({"src": sources, "sent": sent}).groupby("src")["sent"].agg(["sum", "count"])
    window = max(window_seconds or RATE_WINDOW_SECONDS, 0.5)
    return {
        str(src): {"pps": float(row["sum"]) / window, "fps": float(row["count"]) / window}
        for src, row in grouped.iterrows()
    }


def remote_endpoint(src_ip: str, dst_ip: str) -> str:
    """Extremo remoto de un flujo: el origen, salvo que el origen sea el propio servidor protegido."""
    if src_ip in PROTECTED_IPS and dst_ip not in PROTECTED_IPS:
        return dst_ip
    return src_ip


def is_rate_attack_source(source_rate: dict[str, float]) -> bool:
    return source_rate["pps"] >= RATE_ATTACK_PPS or source_rate["fps"] >= RATE_ATTACK_FLOWS_PER_SECOND


def rate_risk_score(source_rate: dict[str, float]) -> int:
    ratio = max(
        source_rate["pps"] / max(RATE_ATTACK_PPS, 1),
        source_rate["fps"] / max(RATE_ATTACK_FLOWS_PER_SECOND, 1),
    )
    return min(100, round(ratio * 65))


def is_non_unicast_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_multicast or getattr(address, "is_broadcast", False) or str(address).endswith(".255")


def normalize_cicflowmeter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adapta la salida snake_case de cicflowmeter Python al CIC-DDoS2019."""
    python_format = "flow_duration" in df.columns
    if not python_format:
        return df

    normalized = df.rename(columns=PY_CICFLOWMETER_COLUMNS).copy()
    # El paquete Python usa segundos; CIC-DDoS2019 expresa tiempos en microsegundos.
    for column in PY_CICFLOWMETER_TIME_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce") * 1_000_000
    return normalized


def run_block_script(ip: str, action: str = "block") -> bool:
    """Ejecuta el script de firewall (block/unblock) para una IP."""
    validate_ip(ip)
    try:
        if platform.system() == "Windows":
            args = [BLOCK_SCRIPT, ip] if action == "block" else [BLOCK_SCRIPT, action, ip]
            result = subprocess.run(args, shell=True, check=False, timeout=8)
        else:
            command = [BLOCK_SCRIPT, action, ip]
            if BLOCK_USE_SUDO and getattr(os, "geteuid", lambda: 1)() != 0:
                command = ["sudo", "-n", *command]
            result = subprocess.run(command, check=False, timeout=8, capture_output=True, text=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        state.last_error = f"No se pudo ejecutar el script de firewall ({action} {ip}): {exc}"
        return False
    if result.returncode != 0:
        state.last_error = (
            f"No se pudo aplicar '{action}' a {ip}; el script de firewall termino con codigo {result.returncode}."
        )
        return False
    return True


def apply_block(ip: str, reason: str, risk: int, source: str, **details: Any) -> None:
    """Registra la IP y, si AUTO_BLOCK=true, aplica la regla temporal en el firewall."""
    detected_at = time.time()
    if ip in PROTECTED_IPS:
        log_mitigation({"evento": "bloqueo_omitido_ip_protegida", "ip": ip, "origen": source, "riesgo": risk})
        return
    if not state.add_block(ip, reason, risk, source):
        return
    record: dict[str, Any] = {
        "evento": "bloqueo" if AUTO_BLOCK else "deteccion",
        "ip": ip,
        "origen": source,
        "riesgo": risk,
        "bloqueo_automatico": AUTO_BLOCK,
        "ttl_s": BLOCK_TTL_SECONDS,
        "t_deteccion_epoch": detected_at,
        **details,
    }
    if AUTO_BLOCK:
        started = time.perf_counter()
        ok = run_block_script(ip, "block")
        record["firewall_ok"] = ok
        record["firewall_ms"] = round(1000 * (time.perf_counter() - started), 3)
        record["t_bloqueo_epoch"] = time.time()
        state.set_block_status(ip, "Bloqueada" if ok else "Error de bloqueo")
    log_mitigation(record)


def block_expiry_loop() -> None:
    """Retira los bloqueos vencidos del estado y del firewall (reglas temporales)."""
    while True:
        try:
            for item in state.pop_expired_blocks():
                firewall_ok = run_block_script(item["ip"], "unblock") if item["status"] == "Bloqueada" else None
                log_mitigation({"evento": "caducidad", "ip": item["ip"], "firewall_ok": firewall_ok,
                                "duracion_s": round(time.time() - item["sinceEpoch"], 3)})
        except Exception as exc:
            state.last_error = str(exc)
        time.sleep(1)


def log_mitigation(record: dict[str, Any]) -> None:
    """Bitacora auditable de detecciones, bloqueos y desbloqueos (una linea JSON por evento)."""
    record = {"t_registro_epoch": time.time(), "hora": datetime.now().isoformat(timespec="milliseconds"), **record}
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with MITIGATION_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        state.last_error = f"No se pudo escribir la bitacora de mitigacion: {exc}"


def is_whitelisted(ip: str) -> bool:
    if ip in WHITELIST_IPS:
        return True
    try:
        addr = ipaddress.ip_address(ip)
        for entry in WHITELIST_IPS:
            if "/" in entry and addr in ipaddress.ip_network(entry, strict=False):
                return True
    except ValueError:
        return False
    return False


def validate_ip(ip: str) -> None:
    try:
        ipaddress.ip_address(ip)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="IP invalida") from exc


def pick_value(row: dict[str, Any], names: list[str], default: Any) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return default


def safe_number(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def shell_quote(value: str) -> str:
    if platform.system() == "Windows":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)
