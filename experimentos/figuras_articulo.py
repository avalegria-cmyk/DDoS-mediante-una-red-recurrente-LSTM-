"""Genera las figuras del articulo a partir de los resultados guardados.

  figura_mitigacion_linea_tiempo.png  peticiones respondidas por segundo (atacante y clientes legitimos)
                                      en una corrida de la inundacion HTTP, a partir del registro de la victima
  figura_rf_carga.png                 latencia por flujo y throughput del Random Forest desplegado

Uso: python3 experimentos/figuras_articulo.py [--log logs/victima_8080.log] [--run mitigacion_multi_http_rep3.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experimentos" / "resultados"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorias 1 y 2 de la paleta, en orden fijo
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "text.color": INK})


def clean(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def timeline(log: Path, run: Path) -> dict:
    d = json.loads(run.read_text())
    t0, attacker, clients = d["t_inicio_ataque"], d["atacante"], set(d["clientes"])
    block, end = d["s_ataque_a_bloqueo"], d.get("s_fin_ataque", d["duracion_ataque_s"])
    lo, hi = -20, int(end) + 20
    pat = re.compile(r'^(\S+) - - \[(\d+)/(\w+)/(\d+) (\d+):(\d+):(\d+)\] "GET')
    months = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
    legit, attack = Counter(), Counter()
    with log.open(errors="ignore") as fh:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            ip = m.group(1)
            if ip != attacker and ip not in clients:
                continue
            ts = dt.datetime(int(m.group(4)), months[m.group(3)], int(m.group(2)), int(m.group(5)), int(m.group(6)), int(m.group(7))).timestamp()
            sec = int(ts - int(t0))
            if lo <= sec < hi:
                (attack if ip == attacker else legit)[sec] += 1
    xs = list(range(lo, hi))
    series = {"segundo": xs, "legitimas_respondidas": [legit[x] for x in xs], "atacante_respondidas": [attack[x] for x in xs],
              "s_bloqueo": block, "s_fin_ataque": end, "corrida": run.name}
    (RES / "figura_mitigacion_linea_tiempo.json").write_text(json.dumps(series, indent=1))

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.9), sharex=True, gridspec_kw={"hspace": 0.28})
    for ax, key, color, title in ((axes[0], "atacante_respondidas", ORANGE, "Attacker: requests answered per second"),
                                  (axes[1], "legitimas_respondidas", BLUE, "Five legitimate clients: requests answered per second")):
        ax.axvspan(0, end, color="#f3f2ee", zorder=0)
        ax.plot(xs, series[key], color=color, linewidth=2, solid_capstyle="round")
        ax.axvline(block, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
        ax.set_ylim(bottom=0)
        clean(ax)
    top = axes[0].get_ylim()[1]
    axes[0].text(block + 1, top * 0.92, f"rule active at {block:.1f} s", fontsize=8, color=INK, va="top")
    axes[0].text((block + end) / 2 + 1, top * 0.55, "attack in progress\n(0 to 45 s)", fontsize=8, color=MUTED, ha="center", va="center")
    axes[1].set_xlabel("Time since the first attack packet (s)")
    fig.savefig(RES / "figura_mitigacion_linea_tiempo.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return series


def load_figure() -> None:
    lot = json.loads((RES / "carga_rf.json").read_text())["lotes"]
    x = [r["flujos_por_lote"] for r in lot]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"wspace": 0.32})
    for ax, key, title in ((axes[0], "latencia_por_flujo_ms", "Latency per flow (ms)"), (axes[1], "throughput_flujos_s", "Throughput (flows per second)")):
        ax.plot(x, [r[key] for r in lot], color=BLUE, linewidth=2, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
        ax.set_xlabel("Flows per batch")
        clean(ax)
    a, b = lot[0], lot[-2]
    axes[0].annotate(f"{a['latencia_por_flujo_ms']:.1f} ms", (x[0], a["latencia_por_flujo_ms"]), textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK)
    axes[0].annotate(f"{b['latencia_por_flujo_ms']:.4f} ms", (x[-2], b["latencia_por_flujo_ms"]), textcoords="offset points", xytext=(2, 13), fontsize=8, color=INK)
    axes[1].annotate(f"{b['throughput_flujos_s']:,.0f}", (x[-2], b["throughput_flujos_s"]), textcoords="offset points", xytext=(-8, -17), fontsize=8, color=INK)
    fig.savefig(RES / "figura_rf_carga.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "victima_8080.log")
    ap.add_argument("--run", type=Path, default=RES / "mitigacion_multi_http_rep3.json")
    args = ap.parse_args()
    load_figure()
    if args.log.exists():
        s = timeline(args.log, args.run)
        import statistics as st
        xs = s["segundo"]; lg = s["legitimas_respondidas"]; at = s["atacante_respondidas"]
        seg = lambda a, b, v: [v[i] for i, x in enumerate(xs) if a <= x < b]
        print("legitimas/s antes:", round(st.mean(seg(-20, 0, lg)), 1), "| ataque sin bloqueo:", round(st.mean(seg(1, int(s["s_bloqueo"]), lg)), 1),
              "| ataque con bloqueo:", round(st.mean(seg(int(s["s_bloqueo"]) + 2, 45, lg)), 1), "| despues:", round(st.mean(seg(46, 65, lg)), 1))
        print("atacante/s sin bloqueo:", round(st.mean(seg(1, int(s["s_bloqueo"]), at))), "| con bloqueo:", round(st.mean(seg(int(s["s_bloqueo"]) + 2, 45, at)), 1))
