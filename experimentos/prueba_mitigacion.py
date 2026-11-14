"""Prueba de mitigacion en la misma maquina (correccion B).

Requiere el detector levantado con captura y bloqueo reales:
    sudo AUTO_BLOCK=true BLOCK_TTL_SECONDS=60 CAPTURE_INTERFACE=lo0 KEEP_CAPTURE_FILES=true \
        ./scripts/iniciar_admin_mac_rf11.sh

Fases:
  1. Solo trafico legitimo (peticiones HTTP espaciadas a 127.0.0.1:8080).
  2. Inundacion HTTP (hilos que abren una conexion nueva por peticion) contra la IP local de la maquina,
     con trafico legitimo en paralelo.
  3. Sondeo de la victima desde la IP atacante hasta que el bloqueo caduca.

Mide: tiempo desde el inicio del ataque hasta el bloqueo, efecto real del bloqueo y su duracion.
Uso: python3 experimentos/prueba_mitigacion.py [--legit-seconds 50] [--attack-seconds 20] [--concurrency 50]
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:4100"


def get_json(path: str):
    with urllib.request.urlopen(API + path, timeout=3) as r:
        return json.loads(r.read())


def probe(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def http_flood(ip: str, port: int, seconds: float, threads: int, counters: dict) -> list[threading.Thread]:
    """Cada hilo abre una conexion TCP nueva por peticion, sin pausa, durante `seconds`."""
    request = f"GET / HTTP/1.0\r\nHost: {ip}\r\n\r\n".encode()
    end = time.time() + seconds

    def worker() -> None:
        while time.time() < end:
            try:
                with socket.create_connection((ip, port), timeout=1.0) as conn:
                    conn.sendall(request)
                    while conn.recv(65536):
                        pass
                counters["ok"] += 1
            except OSError:
                counters["fallo"] += 1

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for w in workers:
        w.start()
    return workers


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # no envia nada; solo elige la interfaz de salida
        return s.getsockname()[0]
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legit-seconds", type=int, default=50)
    ap.add_argument("--attack-seconds", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "experimentos" / "resultados" / "mitigacion.json")
    args = ap.parse_args()

    ip = lan_ip()
    victim_local, victim_lan = "http://127.0.0.1:8080/", f"http://{ip}:8080/"
    health = get_json("/api/health")
    assert health["auto_block"], "El detector debe arrancar con AUTO_BLOCK=true"
    log: dict = {"ip_atacante": ip, "ttl_s": health["block_ttl_seconds"], "captura_s": get_json("/api/capture/status")["captureSeconds"],
                 "concurrencia": args.concurrency, "ataque_s": args.attack_seconds}
    stop = threading.Event()
    legit = {"ok": 0, "fallo": 0}

    def legit_loop() -> None:
        while not stop.is_set():
            legit["ok" if probe(victim_local, 2.0) else "fallo"] += 1
            time.sleep(0.5)

    threading.Thread(target=legit_loop, daemon=True).start()

    print(f"Fase 1: {args.legit_seconds} s de trafico legitimo")
    log["t_inicio_legitimo"] = time.time()
    time.sleep(args.legit_seconds)
    log["bloqueos_tras_fase_legitima"] = get_json("/api/blocked-ips")["blockedIps"]
    log["legitimo_fase1"] = dict(legit)

    print(f"Fase 2: inundacion HTTP {args.attack_seconds} s, concurrencia {args.concurrency}, contra {victim_lan}")
    log["t_inicio_ataque"] = t_attack = time.time()
    flood = {"ok": 0, "fallo": 0}
    workers = http_flood(ip, 8080, args.attack_seconds, args.concurrency, flood)

    probes: list[tuple[float, bool]] = []
    t_block_api = t_drop = t_back = None
    deadline = t_attack + args.attack_seconds + 240
    while time.time() < deadline:
        now = time.time()
        blocked = [b for b in get_json("/api/blocked-ips")["blockedIps"] if b["ip"] == ip]
        if blocked and t_block_api is None:
            t_block_api = now
            log["bloqueo_api"] = blocked[0]
            print(f"  bloqueo registrado a los {now - t_attack:.1f} s del inicio del ataque")
        ok = probe(victim_lan)
        probes.append((round(now - t_attack, 2), ok))
        if not ok and t_block_api is not None and t_drop is None:
            t_drop = now
        if ok and t_drop is not None and t_back is None and not blocked:
            t_back = now
            print(f"  la victima vuelve a responder al atacante a los {now - t_attack:.1f} s")
            break
        time.sleep(0.3)

    for w in workers:
        w.join(timeout=5)
    stop.set()
    log.update(
        t_bloqueo_visto_api=t_block_api, t_victima_inaccesible=t_drop, t_victima_recuperada=t_back,
        s_ataque_a_bloqueo=None if t_block_api is None else round(t_block_api - t_attack, 2),
        s_bloqueo_efectivo=None if not (t_drop and t_back) else round(t_back - t_drop, 2),
        sondeos=probes, legitimo_total=dict(legit),
        ataque_peticiones=dict(flood),
        salud_final=get_json("/api/health"), captura_final=get_json("/api/capture/status"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: log[k] for k in ("s_ataque_a_bloqueo", "s_bloqueo_efectivo", "legitimo_total", "ataque_peticiones")}, indent=2, ensure_ascii=False))
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
