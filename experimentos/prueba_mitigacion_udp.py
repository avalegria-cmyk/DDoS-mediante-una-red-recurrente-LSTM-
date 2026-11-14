"""Inundacion UDP de un solo flujo contra la victima local; mide el tiempo hasta el bloqueo.

Requiere el detector levantado con AUTO_BLOCK=true (ver prueba_mitigacion.py).
Uso: python3 experimentos/prueba_mitigacion_udp.py [--seconds 20] [--out archivo.json]
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "experimentos" / "resultados" / "mitigacion_udp.json")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("192.0.2.1", 9))
    ip = s.getsockname()[0]
    s.close()

    def blocked() -> list:
        with urllib.request.urlopen("http://127.0.0.1:4100/api/blocked-ips", timeout=3) as r:
            return [b for b in json.loads(r.read())["blockedIps"] if b["ip"] == ip]

    while blocked():  # esperar a que caduque cualquier bloqueo anterior
        time.sleep(2)
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.bind((ip, 40123))
    payload = b"A" * 400
    t0, sent, t_block, last_poll = time.time(), 0, None, 0.0
    while time.time() - t0 < args.seconds:
        for _ in range(30):
            try:
                u.sendto(payload, (ip, 8080)); sent += 1
            except OSError:
                pass
        time.sleep(0.01)
        if t_block is None and time.time() - last_poll > 0.3:
            last_poll = time.time()
            if blocked():
                t_block = time.time()
    while t_block is None and time.time() - t0 < 120:
        if blocked():
            t_block = time.time()
        time.sleep(0.3)
    result = {"ataque": "inundacion UDP de un solo flujo (400 bytes)", "ip_atacante": ip, "t_inicio_ataque": t0,
              "duracion_s": args.seconds, "datagramas": sent, "pps": round(sent / args.seconds),
              "s_ataque_a_bloqueo": None if t_block is None else round(t_block - t0, 2),
              "bloqueo_api": (blocked() or [None])[0]}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("pps", "s_ataque_a_bloqueo")}))


if __name__ == "__main__":
    main()
