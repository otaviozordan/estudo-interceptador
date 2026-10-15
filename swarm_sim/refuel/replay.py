"""Replay de episodios de reabastecimento: grava quadros de um episodio e desenha/anima no mapa.

Mapa: relevo, rotas (tracejadas), zonas prioritarias, radares, bases; aeronaves (verdade, triangulos), pistas de
radar (quadrado ciano = julgada legitima pelo metodo, x cinza = julgada falso alarme), drones (circulos) com linha
ate o alvo (laranja = transito pela pista de radar, magenta = rendezvous com o sensor proprio). Eventos: estrela
verde = reabastecida, X vermelho = caiu. Janela: espaco play/pausa, <-/-> +-10 s, R reiniciar.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import patheffects as pe
from matplotlib.colors import LightSource
from matplotlib.patches import Circle

from .env import RefuelEnv

DR_COLOR = {0: "#9e9e9e", 1: "#ff9800", 4: "#e040fb", 2: "#2e7d32", 3: "#b71c1c"}
DR_LABEL = {0: "na base", 1: "transito (radar)", 4: "rendezvous (sensor)", 2: "reabasteceu", 3: "sem autonomia"}


def record(env: RefuelEnv, policy) -> tuple[list[dict], float]:
    """Roda o episodio guardando um quadro por passo do mundo (com a decisao legitima/falsa do metodo)."""
    frames: list[dict] = []
    step_world = env._step_world
    classify = getattr(policy, "classify", None)

    def snap():
        pic = env._pic
        ok = (classify(env, pic) >= 0.5) if classify is not None and pic else np.ones(len(pic), bool)
        truth = np.array([p["truth"] is not None for p in pic], bool)
        frames.append({
            "t": env.t,
            "ac": np.array([b.x[0:2] for b in env.bodies]), "ac_status": env.ac_status.copy(),
            "tracks": np.array([[p["n"], p["e"]] for p in pic]).reshape(-1, 2), "tr_ok": ok, "tr_truth": truth,
            "dr": env.dr_pos.copy(),
            "dr_state": np.where((env.dr_state == 1) & (env.dr_lock >= 0), 4, env.dr_state),
            "dr_aim": np.array([env.bodies[env.dr_lock[d]].x[0:2] if env.dr_lock[d] >= 0 else
                                env.dr_last[d, :2] if env.dr_state[d] == 1 else env.dr_pos[d]
                                for d in range(env.n_dr)]).reshape(-1, 2),
            "info": dict(env.info)})

    def wrapped():
        r = step_world()
        snap()
        return r

    env._step_world = wrapped
    snap()
    ret = 0.0
    while not env.done:
        _, r, _ = env.step(policy(env))
        ret += r
    env._step_world = step_world
    return frames, ret


def _figure(env: RefuelEnv, frames: list[dict], title: str, judged: bool):
    """Monta a figura; devolve (fig, draw(i, paused))."""
    sc, m = env.sc, env.m
    T = sc.terrain
    ext = T.extent                                              # (n0, n1, e0, e1)
    fig, ax = plt.subplots(figsize=(10, 9))
    shade = LightSource(315, 45).shade(T.z, cmap=plt.cm.gist_earth, vert_exag=2, blend_mode="soft")
    ax.imshow(shade, origin="lower", extent=(ext[2], ext[3], ext[0], ext[1]), alpha=0.85)
    for p in env.paths:
        ax.plot(p.P[:, 1], p.P[:, 0], "--", color="w", lw=1, alpha=0.7)
        if not p.loop:
            ax.plot(p.impact_ne[1], p.impact_ne[0], "v", color="k", ms=7)
    for z in m.zones:
        ax.add_patch(Circle((z.e, z.n), z.radius, fc="#e53935", ec="#b71c1c", alpha=0.25, lw=1.5))
    for r in m.radars:
        ax.plot(r.e, r.n, "^", color="#1565c0", ms=9, mec="w")
    for b in m.bases:
        ax.plot(b.e, b.n, "s", color="#ff9800", ms=11, mec="k")
        ax.annotate(f"{b.drones}", (b.e, b.n), xytext=(6, 6), textcoords="offset points", color="k", weight="bold")
    ax.set_xlim(ext[2], ext[3])
    ax.set_ylim(ext[0], ext[1])
    ax.set_xlabel("leste [m]")
    ax.set_ylabel("norte [m]")

    ac_sc = ax.scatter([], [], marker="^", s=90, c="#fdd835", edgecolors="k", zorder=5, label="aeronave (verdade)")
    tr_ln, = ax.plot([], [], "s", ms=9, mfc="none", mec="#00e5ff", mew=2, zorder=8,
                     label="pista julgada legitima" if judged else "pista de radar")
    fa_ln, = ax.plot([], [], "x", ms=7, color="#212121", mew=1.6, zorder=8, label="pista julgada falsa",
                     path_effects=[pe.withStroke(linewidth=3.5, foreground="w")])
    dr_sc = ax.scatter([], [], marker="o", s=40, edgecolors="k", zorder=6, label="drone")
    ok_sc = ax.scatter([], [], marker="*", s=260, c="#00c853", edgecolors="k", zorder=7, label="reabastecida")
    cr_sc = ax.scatter([], [], marker="X", s=160, c="#d50000", edgecolors="k", zorder=7, label="caiu")
    lines = [ax.plot([], [], "-", color="#ff9800", lw=1, alpha=0.8)[0] for _ in range(env.n_dr)]
    trails = [ax.plot([], [], "-", color="#ff9800", lw=0.8, alpha=0.4)[0] for _ in range(env.n_dr)]
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8.5,
                  bbox=dict(fc="white", alpha=0.85))
    handles = [h for h in ax.get_legend_handles_labels()[0] if judged or h is not fa_ln]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.11, top=0.95)
    ax.set_title(title, fontsize=10)
    empty = np.empty((0, 2))
    drpath = np.array([f["dr"] for f in frames]).reshape(len(frames), env.n_dr, 2)
    moved_all = np.linalg.norm(drpath - drpath[:1], axis=2) > 1

    def pts(a, sel):
        return a[sel][:, ::-1] if np.any(sel) else empty

    def draw(i: int, paused: bool = False):
        f = frames[i]
        st = f["ac_status"]
        live = st == 0
        ac_sc.set_offsets(pts(f["ac"], live))
        ok_sc.set_offsets(pts(f["ac"], st == 1))
        cr_sc.set_offsets(pts(f["ac"], st == 2))
        tr_ln.set_data(f["tracks"][f["tr_ok"], 1], f["tracks"][f["tr_ok"], 0])
        fa_ln.set_data(f["tracks"][~f["tr_ok"], 1], f["tracks"][~f["tr_ok"], 0])
        dr_sc.set_offsets(f["dr"][:, ::-1] if len(f["dr"]) else empty)
        dr_sc.set_facecolors([DR_COLOR[s] for s in f["dr_state"]])
        for d, ln in enumerate(lines):
            ln.set_color(DR_COLOR[f["dr_state"][d]])
            if f["dr_state"][d] in (1, 4):
                ln.set_data([f["dr"][d, 1], f["dr_aim"][d, 1]], [f["dr"][d, 0], f["dr_aim"][d, 0]])
            else:
                ln.set_data([], [])
            path, moved = drpath[: i + 1, d], moved_all[: i + 1, d]
            trails[d].set_data(path[moved, 1], path[moved, 0])
        inf = f["info"]
        cnt = {k: int(np.sum(f["dr_state"] == k)) for k in DR_LABEL}
        ok, y = f["tr_ok"], f["tr_truth"]
        leg = (f"pistas: {len(ok)} ({int(ok.sum())} legitimas, {int((~ok).sum())} falsas; "
               f"{int(np.sum(ok != y))} erradas)" if judged else f"pistas de radar: {len(ok)}")
        hud.set_text(f"t = {f['t']:6.0f} s {'(pausa)' if paused else ''}\n"
                     f"aeronaves: {int(live.sum())} na lista, {inf['refueled']} reabastecidas, "
                     f"{inf['crashed']} caidas ({inf['crashed_priority']} prior.)\n"
                     f"{leg}   trocas de alvo em voo: {inf.get('retasks', 0)}\n"
                     f"drones: " + ", ".join(f"{cnt[k]} {DR_LABEL[k]}" for k in DR_LABEL))
        return [ac_sc, tr_ln, fa_ln, dr_sc, ok_sc, cr_sc, hud, *lines, *trails]

    return fig, draw


def save_gif(env: RefuelEnv, frames: list[dict], path: str, title: str, judged: bool, speed: float = 20.0):
    """Grava o episodio (``speed`` segundos simulados por segundo de video, 10 quadros/s)."""
    fig, draw = _figure(env, frames, title, judged)
    stride = max(1, int(round(speed * 0.1 / env.cfg.dt)))
    idx = list(range(0, len(frames), stride)) + [len(frames) - 1] * 20
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    FuncAnimation(fig, lambda k: draw(idx[k]), frames=len(idx), interval=100, blit=False).save(path, fps=10, dpi=72)
    plt.close(fig)


def show(env: RefuelEnv, frames: list[dict], title: str, judged: bool, speed: float = 10.0):
    """Janela interativa."""
    fig, draw = _figure(env, frames, title, judged)
    state = {"i": 0, "play": True}
    stride = max(1, int(round(speed * 0.1 / env.cfg.dt)))
    jump = int(10 / env.cfg.dt)

    def tick(_):
        if state["play"]:
            state["i"] = min(state["i"] + stride, len(frames) - 1)
        return draw(state["i"], not state["play"])

    def key(ev):
        if ev.key == " ":
            state["play"] = not state["play"]
        elif ev.key == "right":
            state["i"] = min(state["i"] + jump, len(frames) - 1)
        elif ev.key == "left":
            state["i"] = max(state["i"] - jump, 0)
        elif ev.key == "r":
            state["i"] = 0
        draw(state["i"], not state["play"])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", key)
    anim = FuncAnimation(fig, tick, interval=100, blit=False, cache_frame_data=False)  # noqa: F841 (manter referencia)
    plt.show()
