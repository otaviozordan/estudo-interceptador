"""Mapa estatico de uma missao (PNG salvo ao lado do JSON): relevo, cobertura dos radares, rotas com as areas de
sorteio do preset, pontos de impacto, zonas prioritarias, bases e radares."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .mission import Mission
from .radar import coverage_mask
from .terrain import Terrain

AC_COLORS = ["#fdd835", "#ff7043", "#ab47bc", "#26c6da", "#9ccc65", "#ec407a"]


def save_mission_map(m: Mission, terrain: Terrain, path: str, coverage_agl: float | None = 120.0, cells: int = 70):
    """Grava o mapa da missao em `path` (PNG). coverage_agl=None nao calcula a cobertura dos radares."""
    fig = Figure(figsize=(9, 9.6))
    ax = fig.add_subplot(111)
    n0, n1, e0, e1 = terrain.extent
    shade = LightSource(315, 45).shade(terrain.z, cmap=_cmap(), vert_exag=2, blend_mode="soft")
    ax.imshow(shade, origin="lower", extent=(e0, e1, n0, n1), alpha=0.9)
    radars = [r for r in m.radars if r.enabled]
    if coverage_agl is not None and radars:
        cov = None
        for r in radars:
            ns, es, mk = coverage_mask(terrain, r, coverage_agl, cells=cells)
            cov = mk if cov is None else cov | mk
        ax.contourf(es, ns, cov.astype(float), levels=[0.5, 1.5], colors="none", hatches=["...."], zorder=2)
        ax.contour(es, ns, cov.astype(float), levels=[0.5], colors="#1565c0", linewidths=0.8, zorder=2)
    for z in m.zones:
        ax.add_patch(Circle((z.e, z.n), z.radius, fc="#e53935", ec="#b71c1c", alpha=0.28, lw=1.5, zorder=3))
        ax.annotate(z.name, (z.e, z.n), ha="center", va="center", fontsize=8, color="#b71c1c", weight="bold", zorder=3)
    for k, a in enumerate(m.aircraft):
        c = AC_COLORS[k % len(AC_COLORS)]
        pts = ([(a.start["n"], a.start["e"])] if a.start else []) + [(w.n, w.e) for w in a.waypoints]
        if pts:
            P = np.array(pts)
            ax.plot(P[:, 1], P[:, 0], "--", color=c, lw=1.6, zorder=4, label=f"{a.name}")
        if a.start:
            ax.plot(a.start["e"], a.start["n"], "o", color=c, mec="k", ms=8, zorder=5)
            if a.start_radius > 0:
                ax.add_patch(Circle((a.start["e"], a.start["n"]), a.start_radius, fc=c, ec=c, alpha=0.15, zorder=3))
        for w in a.waypoints:
            if w.pos_radius > 0:
                ax.add_patch(Circle((w.e, w.n), w.pos_radius, fc=c, ec=c, alpha=0.15, ls="--", zorder=3))
            if w.impact:
                ax.plot(w.e, w.n, "v", color=c, mec="k", ms=10, zorder=5)
            else:
                ax.plot(w.e, w.n, ".", color=c, ms=6, zorder=5)
    for r in radars:
        ax.plot(r.e, r.n, "^", color="#1565c0", mec="w", ms=11, zorder=6)
        ax.annotate(r.name, (r.e, r.n), xytext=(6, -10), textcoords="offset points", fontsize=8, color="#0d47a1")
    for b in m.bases:
        ax.plot(b.e, b.n, "s", color="#ff9800", mec="k", ms=12, zorder=6)
        ax.annotate(f"{b.name} ({b.drones})", (b.e, b.n), xytext=(7, 6), textcoords="offset points", fontsize=8,
                    weight="bold")
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_xlabel("leste [m]")
    ax.set_ylabel("norte [m]")
    rf = m.refuel
    preset = "preset aleatorio" if m.is_preset else "fixa"
    ax.set_title(f"{m.name} | {len(m.aircraft)} aeronave(s), {len(radars)} radar(es), {len(m.bases)} base(s), "
                 f"{sum(b.drones for b in m.bases)} drones\n{preset}, {rf.runs} execucoes | "
                 f"origem {m.origin['lat']:.4f}, {m.origin['lon']:.4f} ({m.origin['size_km']:g} km)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.5, 0.015, "▲ radar (pontilhado = cobertura a %s m AGL)   ■ base (drones)   ● inicio   ▼ impacto   "
             "circulos claros = area de sorteio   vermelho = zona prioritaria"
             % (f"{coverage_agl:g}" if coverage_agl is not None else "-"), ha="center", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.08)
    fig.savefig(path, dpi=90)


def _cmap():
    import matplotlib
    return matplotlib.colormaps["gist_earth"]
