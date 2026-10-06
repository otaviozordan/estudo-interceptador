"""Graficos: mapa com hillshade + trajetorias, perfis AGL e atitude."""
from __future__ import annotations

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def plot_map(sim, path: str | None = None, show_routes: bool = True):
    T = sim.terrain
    res = sim.results()
    fig, ax = plt.subplots(figsize=(9, 8))
    ext = [T.e0 / 1000, T.e_max / 1000, T.n0 / 1000, T.n_max / 1000]
    ax.imshow(T.z, extent=ext, origin="lower", cmap="terrain", alpha=0.9)
    ax.imshow(T.hillshade(), extent=ext, origin="lower", cmap="gray", alpha=0.35)
    for k, a in enumerate(sim.agents):
        c, r = COLORS[k % len(COLORS)], res[a.name]
        if show_routes:
            w = np.array([[q.n, q.e] for q in a.navigator.wps])
            if a.navigator.route.loop:
                w = np.vstack([w, w[:1]])
            ax.plot(w[:, 1] / 1000, w[:, 0] / 1000, "--", color=c, lw=0.8, alpha=0.8)
            ax.plot(w[:, 1] / 1000, w[:, 0] / 1000, "o", color=c, ms=3)
        ax.plot(r["e"] / 1000, r["n"] / 1000, color=c, lw=1.6, label=f"{a.name} ({a.navigator.route.kind})")
        if len(r["e"]):
            ax.plot(r["e"][0] / 1000, r["n"][0] / 1000, "^", color=c, mec="k")
        if a.crash_pos:
            ax.plot(a.crash_pos[1] / 1000, a.crash_pos[0] / 1000, "X", color="k", ms=13, mec="w")
    ax.set_xlabel("Leste [km]")
    ax.set_ylabel("Norte [km]")
    ax.set_title("Simulacao sobre DEM (Copernicus GLO-30)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_profiles(sim, path: str | None = None):
    res = sim.results()
    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for k, a in enumerate(sim.agents):
        c, r = COLORS[k % len(COLORS)], res[a.name]
        axs[0].plot(r["t"], r["alt"], color=c, label=a.name)
        axs[0].plot(r["t"], r["terrain"], color=c, ls=":", lw=1)
        axs[1].plot(r["t"], r["agl"], color=c)
        axs[2].plot(r["t"], r["Va"], color=c)
        axs[3].plot(r["t"], np.degrees(r["phi"]), color=c, lw=0.8)
    axs[0].set_ylabel("Altitude MSL [m]\n(pontilhado: terreno)")
    axs[1].set_ylabel("AGL [m]")
    axs[1].axhline(0, color="k", lw=0.8)
    axs[2].set_ylabel("Va [m/s]")
    axs[3].set_ylabel("Rolagem [deg]")
    axs[3].set_xlabel("t [s]")
    axs[0].legend(ncol=3, fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_3d(sim, path: str | None = None, stride: int = 6):
    T = sim.terrain
    res = sim.results()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    s = max(1, int(round(300 / T.res)))
    E, N = np.meshgrid((T.e0 + T.res * np.arange(T.nx))[::s] / 1000,
                       (T.n0 + T.res * np.arange(T.ny))[::s] / 1000)
    ax.plot_surface(E, N, T.z[::s, ::s], cmap="terrain", alpha=0.6, linewidth=0, antialiased=False)
    for k, a in enumerate(sim.agents):
        r = res[a.name]
        ax.plot(r["e"][::stride] / 1000, r["n"][::stride] / 1000, r["alt"][::stride],
                color=COLORS[k % len(COLORS)], lw=1.8, label=a.name)
    ax.set_xlabel("Leste [km]"); ax.set_ylabel("Norte [km]"); ax.set_zlabel("Alt [m]")
    ax.set_zlim(T.z.min(), T.z.max() + 600)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
