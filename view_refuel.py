"""Replay de um episodio de reabastecimento (ambiente cinematico de swarm_sim/refuel).

    python view_refuel.py missions/.../missao.json                         # politica hungaro_prioridade
    python view_refuel.py missions/.../missao.json --policy results/refuel/il_ppo.pt --seed 3
    python view_refuel.py missions/.../missao.json --policy guloso_proximo --save results/refuel/ep.gif

--policy: nome de uma linha de base (aleatorio, guloso_proximo, guloso_urgente, guloso_prioridade, hungaro,
hungaro_prioridade) ou pesos .pt salvos por run_refuel.py (il.pt, ppo.pt, il_ppo.pt).
Mapa: relevo, rotas (tracejadas), zonas prioritarias, radares, bases; aeronaves (verdade, triangulos),
pistas de radar (quadrados), drones (circulos) com linha ate o alvo (laranja = transito pela
pista de radar, magenta = rendezvous com o sensor proprio). Eventos: estrela verde =
reabastecida, X vermelho = caiu. Teclas: espaco play/pausa, <-/-> +-10 s, R reiniciar.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
from matplotlib.patches import Circle

from swarm_sim.refuel.baselines import BASELINES
from swarm_sim.refuel.env import RefuelConfig, RefuelEnv, Scenario

DR_COLOR = {0: "#9e9e9e", 1: "#ff9800", 4: "#e040fb", 2: "#2e7d32", 3: "#b71c1c"}
DR_LABEL = {0: "na base", 1: "transito (radar)", 4: "rendezvous (sensor)", 2: "reabasteceu", 3: "sem autonomia"}


def make_policy(spec: str):
    if spec in BASELINES:
        return BASELINES[spec]
    import torch
    from swarm_sim.refuel.learn import AllocNet, net_policy
    net = AllocNet()
    net.load_state_dict(torch.load(spec, map_location="cpu"))
    return net_policy(net)


def record(env: RefuelEnv, policy) -> tuple[list[dict], float]:
    """Roda o episodio guardando um quadro por passo do mundo."""
    frames: list[dict] = []
    step_world = env._step_world

    def snap():
        frames.append({
            "t": env.t,
            "ac": np.array([b.x[0:2] for b in env.bodies]), "ac_status": env.ac_status.copy(),
            "tracks": np.array([[p["n"], p["e"]] for p in env._pic]).reshape(-1, 2),
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
    return frames, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission")
    ap.add_argument("--policy", default="hungaro_prioridade")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--contact-radius", type=float, help="sobrescreve refuel.contact_radius da missao [m]")
    ap.add_argument("--speed", type=float, default=10.0, help="segundos simulados por segundo de replay")
    ap.add_argument("--save", help="grava .gif/.mp4 em vez de abrir a janela")
    a = ap.parse_args()

    sc = Scenario.from_file(a.mission, a.synthetic)
    if a.contact_radius is not None:
        sc.m.refuel.contact_radius = a.contact_radius
    env = RefuelEnv(sc, RefuelConfig(), seed=a.seed)
    frames, ret = record(env, make_policy(a.policy))
    s = env.summary()
    print(f"{a.policy} seed {a.seed}: {s['refueled']}/{s['n_aircraft']} reabastecidas, {s['crashed']} caidas "
          f"({s['crashed_priority']} em zona prioritaria), drones sem autonomia {s['drones_failed']}, retorno {ret:.2f}")

    T, m = sc.terrain, env.m
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
    tr_sc = ax.scatter([], [], marker="s", s=60, facecolors="none", edgecolors="#00e5ff", lw=1.5, zorder=4,
                       label="pista de radar")
    dr_sc = ax.scatter([], [], marker="o", s=40, edgecolors="k", zorder=6, label="drone")
    ok_sc = ax.scatter([], [], marker="*", s=260, c="#00c853", edgecolors="k", zorder=7, label="reabastecida")
    cr_sc = ax.scatter([], [], marker="X", s=160, c="#d50000", edgecolors="k", zorder=7, label="caiu")
    lines = [ax.plot([], [], "-", color="#ff9800", lw=1, alpha=0.8)[0] for _ in range(env.n_dr)]
    trails = [ax.plot([], [], "-", color="#ff9800", lw=0.8, alpha=0.4)[0] for _ in range(env.n_dr)]
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9,
                  bbox=dict(fc="white", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{sc.name} | politica {os.path.basename(a.policy)} | seed {a.seed}")

    state = {"i": 0, "play": True}
    stride = max(1, int(round(a.speed * 0.1 / env.cfg.dt)))       # 10 quadros/s de replay

    def draw(i):
        f = frames[i]
        st = f["ac_status"]
        live = st == 0
        ac_sc.set_offsets(f["ac"][live][:, ::-1] if live.any() else np.empty((0, 2)))
        ok_sc.set_offsets(f["ac"][st == 1][:, ::-1] if (st == 1).any() else np.empty((0, 2)))
        cr_sc.set_offsets(f["ac"][st == 2][:, ::-1] if (st == 2).any() else np.empty((0, 2)))
        tr_sc.set_offsets(f["tracks"][:, ::-1] if len(f["tracks"]) else np.empty((0, 2)))
        dr_sc.set_offsets(f["dr"][:, ::-1] if len(f["dr"]) else np.empty((0, 2)))
        dr_sc.set_facecolors([DR_COLOR[s] for s in f["dr_state"]])
        for d, ln in enumerate(lines):
            ln.set_color(DR_COLOR[f["dr_state"][d]])
            if f["dr_state"][d] in (1, 4):
                ln.set_data([f["dr"][d, 1], f["dr_aim"][d, 1]], [f["dr"][d, 0], f["dr_aim"][d, 0]])
            else:
                ln.set_data([], [])
            path = np.array([g["dr"][d] for g in frames[: i + 1]])
            moved = np.linalg.norm(path - path[0], axis=1) > 1
            trails[d].set_data(path[moved, 1], path[moved, 0]) if moved.any() else trails[d].set_data([], [])
        inf = f["info"]
        cnt = {k: int(np.sum(f["dr_state"] == k)) for k in DR_LABEL}
        hud.set_text(f"t = {f['t']:6.0f} s {'' if state['play'] else '(pausa)'}\n"
                     f"aeronaves: {int(live.sum())} na lista, {inf['refueled']} reabastecidas, "
                     f"{inf['crashed']} caidas ({inf['crashed_priority']} prior.)\n"
                     f"pistas de radar: {len(f['tracks'])}\n"
                     f"drones: " + ", ".join(f"{cnt[k]} {DR_LABEL[k]}" for k in DR_LABEL))
        return [ac_sc, tr_sc, dr_sc, ok_sc, cr_sc, hud, *lines, *trails]

    def tick(_):
        if state["play"]:
            state["i"] = min(state["i"] + stride, len(frames) - 1)
        return draw(state["i"])

    def key(ev):
        if ev.key == " ":
            state["play"] = not state["play"]
        elif ev.key == "right":
            state["i"] = min(state["i"] + int(10 / env.cfg.dt), len(frames) - 1)
        elif ev.key == "left":
            state["i"] = max(state["i"] - int(10 / env.cfg.dt), 0)
        elif ev.key == "r":
            state["i"] = 0
        draw(state["i"])
        fig.canvas.draw_idle()

    if a.save:
        idx = list(range(0, len(frames), stride)) + [len(frames) - 1] * 20
        anim = FuncAnimation(fig, lambda k: draw(idx[k]), frames=len(idx), interval=100, blit=False)
        anim.save(a.save, fps=10)
        print(f"salvo em {a.save}")
        return
    fig.canvas.mpl_connect("key_press_event", key)
    anim = FuncAnimation(fig, tick, interval=100, blit=False, cache_frame_data=False)  # noqa: F841 (manter referencia)
    plt.show()


if __name__ == "__main__":
    main()
