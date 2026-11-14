"""Prueba de mitigacion con atacante y varios clientes legitimos de IP distintas (ninguna en lista blanca).

Preparacion (una vez, con contrasena de administrador):
    sudo ./experimentos/red_prueba.sh up
    sudo AUTO_BLOCK=true BLOCK_TTL_SECONDS=60 CAPTURE_INTERFACE=lo0 KEEP_CAPTURE_FILES=true \
        ./scripts/iniciar_admin_mac_rf11.sh

Cada corrida: fase legitima -> ataque sostenido (sigue despues del bloqueo) -> espera a la caducidad.
Mide: tiempo hasta el bloqueo, que clientes legitimos fueron bloqueados y el servicio que reciben
los clientes legitimos antes del ataque, durante el ataque sin bloqueo y durante el ataque con bloqueo.

Uso: python3 experimentos/prueba_mitigacion_multi.py --attack http|udp|slow [--out archivo.json]
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:4100"
VICTIM, ATTACKER, PORT = "10.99.0.1", "10.99.0.2", 8080
# (IP, pausa minima y maxima entre peticiones en segundos): cuatro clientes normales y uno intensivo.
CLIENTS = [("10.99.0.11", 0.8, 2.0), ("10.99.0.12", 0.8, 2.0), ("10.99.0.13", 0.5, 1.5), ("10.99.0.14", 1.0, 3.0), ("10.99.0.15", 0.15, 0.25)]
REQUEST = f"GET / HTTP/1.0\r\nHost: {VICTIM}\r\n\r\n".encode()


def get_json(path: str):
    with urllib.request.urlopen(API + path, timeout=3) as r:
        return json.loads(r.read())


def http_get(src: str, timeout: float = 2.0) -> tuple[bool, float]:
    t0 = time.time()
    try:
        with socket.create_connection((VICTIM, PORT), timeout=timeout, source_address=(src, 0)) as conn:
            conn.sendall(REQUEST)
            data = conn.recv(4096)
            while conn.recv(65536):
                pass
        return data.startswith(b"HTTP/"), time.time() - t0
    except OSError:
        return False, time.time() - t0


def attack_http(end: float, counters: dict, threads: int = 20) -> None:
    def worker() -> None:
        while time.time() < end:
            ok, _ = http_get(ATTACKER, 1.0)
            counters["ok" if ok else "fallo"] += 1
    for _ in range(threads):
        threading.Thread(target=worker, daemon=True).start()


def attack_udp(end: float, counters: dict) -> None:
    def worker() -> None:
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        u.bind((ATTACKER, 40123))
        payload = b"A" * 400
        while time.time() < end:
            for _ in range(30):
                try:
                    u.sendto(payload, (VICTIM, PORT)); counters["ok"] += 1
                except OSError:
                    counters["fallo"] += 1
            time.sleep(0.01)
    threading.Thread(target=worker, daemon=True).start()


def attack_slow(end: float, counters: dict, connections: int = 150) -> None:
    """Ataque de tasa lenta: muchas conexiones abiertas que envian la cabecera a trozos."""
    def worker() -> None:
        socks = []
        while time.time() < end:
            while len(socks) < connections and time.time() < end:
                try:
                    s = socket.create_connection((VICTIM, PORT), timeout=1.0, source_address=(ATTACKER, 0))
                    s.sendall(b"GET / HTTP/1.1\r\nHost: victima\r\n"); socks.append(s); counters["ok"] += 1
                except OSError:
                    counters["fallo"] += 1; break
            alive = []
            for s in socks:
                try:
                    s.sendall(b"X-a: b\r\n"); alive.append(s)
                except OSError:
                    counters["fallo"] += 1
            socks = alive
            time.sleep(3)
        for s in socks:
            try:
                s.close()
            except OSError:
                pass
    threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", choices=["http", "udp", "slow"], required=True)
    ap.add_argument("--legit-seconds", type=float, default=20)
    ap.add_argument("--attack-seconds", type=float, default=45)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    random.seed()

    health = get_json("/api/health")
    assert health["auto_block"], "El detector debe arrancar con AUTO_BLOCK=true"
    while get_json("/api/blocked-ips")["blockedIps"]:  # empezar sin bloqueos vigentes
        time.sleep(2)

    stop = threading.Event()
    requests: list[dict] = []
    lock = threading.Lock()

    def client(ip: str, lo: float, hi: float) -> None:
        while not stop.is_set():
            t = time.time()
            ok, latency = http_get(ip)
            with lock:
                requests.append({"t": t, "cliente": ip, "ok": ok, "latencia": round(latency, 4)})
            time.sleep(random.uniform(lo, hi))

    for ip, lo, hi in CLIENTS:
        threading.Thread(target=client, args=(ip, lo, hi), daemon=True).start()

    t_start = time.time()
    time.sleep(args.legit_seconds)
    blocked_before = get_json("/api/blocked-ips")["blockedIps"]

    t_attack = time.time()
    end = t_attack + args.attack_seconds
    counters = {"ok": 0, "fallo": 0}
    {"http": attack_http, "udp": attack_udp, "slow": attack_slow}[args.attack](end, counters)
    print(f"Ataque {args.attack} desde {ATTACKER} durante {args.attack_seconds:.0f} s")

    t_block = None
    blocked_seen: dict[str, float] = {}
    while time.time() < end + 25:  # se observa hasta 25 s despues de terminar el ataque
        now = time.time()
        for b in get_json("/api/blocked-ips")["blockedIps"]:
            blocked_seen.setdefault(b["ip"], now)
        if t_block is None and ATTACKER in blocked_seen:
            t_block = blocked_seen[ATTACKER]
            print(f"  atacante bloqueado a los {t_block - t_attack:.1f} s")
        time.sleep(0.3)
    stop.set()
    time.sleep(0.5)

    def phase(lo: float, hi: float) -> dict:
        sel = [r for r in requests if lo <= r["t"] < hi]
        ok = [r for r in sel if r["ok"]]
        lat = sorted(r["latencia"] for r in ok)
        return {"peticiones": len(sel), "respondidas": len(ok), "tasa_exito": round(len(ok) / len(sel), 4) if sel else None,
                "latencia_mediana_ms": round(1000 * lat[len(lat) // 2], 2) if lat else None}

    attack_until = (t_block or end)
    result = {
        "ataque": args.attack, "atacante": ATTACKER, "victima": VICTIM, "clientes": [c[0] for c in CLIENTS],
        "t_inicio_ataque": t_attack, "duracion_ataque_s": args.attack_seconds,
        "atacante_bloqueado": t_block is not None,
        "s_ataque_a_bloqueo": None if t_block is None else round(t_block - t_attack, 2),
        "clientes_legitimos_bloqueados": sorted(ip for ip in blocked_seen if ip != ATTACKER),
        "bloqueos_antes_del_ataque": blocked_before,
        "servicio_legitimo": {
            "antes_del_ataque": phase(t_start, t_attack),
            "ataque_sin_bloqueo": phase(t_attack, attack_until),
            "ataque_con_bloqueo": phase(attack_until, end) if t_block is not None and t_block < end else None,
            "despues_del_ataque": phase(end, end + 25),
        },
        "peticiones_por_cliente": {ip: phase_c for ip, phase_c in (
            (c[0], {"total": sum(1 for r in requests if r["cliente"] == c[0]), "ok": sum(1 for r in requests if r["cliente"] == c[0] and r["ok"])}) for c in CLIENTS)},
        "ataque_enviado": counters,
    }
    out = args.out or ROOT / "experimentos" / "resultados" / f"mitigacion_multi_{args.attack}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("s_ataque_a_bloqueo", "clientes_legitimos_bloqueados", "servicio_legitimo")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
