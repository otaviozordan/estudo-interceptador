"""Experimento de guiagem terminal 1 x 1: lei de guiagem x manobra evasiva x aceleracao do drone (Monte Carlo).

    python scripts/experimentos/run_guidance.py   # 300 encontros por combinacao
    python scripts/experimentos/run_guidance.py --realista     # classes das missoes realistas, interceptador 65 m/s
    python scripts/experimentos/run_guidance.py --n 1000 --contact 2

Cada encontro: aeronave em rota reta (25 m/s, +/- variacao) com o perfil evasivo; o drone (42 m/s) comeca no alcance do
sensor proprio (300 m) com aspecto aleatorio (frente, travez, cauda), apontado para a aeronave, e e guiado pela lei em
teste ate o contato (distancia de passagem <= raio de contato) ou ate esgotar o tempo (60 s). Usa as mesmas classes do
ambiente (``refuel/evasion.py``). Grava results/guidance/<carimbo>/guiagem.json e uma tabela em texto.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np                                                  # noqa: E402

from swarm_sim.refuel.env import PathFollower                       # noqa: E402
from swarm_sim.refuel.evasion import closest_approach, guidance_step, make_evasion   # noqa: E402
from swarm_sim.routes import Waypoint                               # noqa: E402

PROFILES = {
    "nenhuma": None,
    "zigue_zague": {"weave": [60, 150], "period": [8, 16], "a_max": 10.0},
    "aleatoria": {"jink": [60, 150], "jink_dt": [2.0, 6.0], "a_max": 10.0},
    "reativa": {"react": 400.0, "react_tgo": 4.0, "react_hold": 3.0, "a_max": 10.0},
    "mista": {"weave": [40, 100], "period": [8, 16], "jink": [30, 80], "react": 400.0, "a_max": 10.0,
              "speed_var": 0.15},
}
LAWS = ("pp", "pn3", "pn4", "apn4", "mpc")

# --realista: as classes de ameaca das missoes realistas (scripts/missoes/make_realista.py: CLASSES) e o
# interceptador de 65 m/s. (evasao, faixa de velocidade [m/s]); o mergulho final nao entra (encontro em 2D).
_FPV = {"jink": [20, 60], "jink_dt": [2.0, 5.0], "a_max": 8.0, "speed_var": 0.1}
CLASSES = {
    "ataque": (None, (42.0, 51.0)),
    "isca": (None, (27.0, 44.0)),
    "vagante": (None, (22.0, 31.0)),
    "fpv": (_FPV, (17.0, 33.0)),
    "fpv_reativo": (_FPV | {"react": 300.0, "react_tgo": 3.0, "react_hold": 2.0}, (17.0, 33.0)),
}


def encounter(law: str, prof, amax: float, contact: float, rng, h: float = 0.05, t_max: float = 60.0,
              v_ac=(22.0, 32.0), v_dr: float = 42.0):
    """Um encontro. Devolve (acertou, tempo ate o contato, menor distancia de passagem, acertou na 1a passagem).
    Passagem = a velocidade de aproximacao troca de sinal a menos de 100 m do alvo."""
    pf = PathFollower([Waypoint(0.0, 0.0), Waypoint(1e5, 0.0)], float(rng.uniform(*v_ac)), loop=False)
    ev = make_evasion(prof, rng)
    s = 2000.0
    rng_r = 300.0
    asp = rng.uniform(0, 2 * np.pi)                                  # aspecto (posicao do drone vista da aeronave)
    pos0, tan, nrm, _ = pf.frame(s)
    ac, acv = pos0.copy(), tan * pf.speed
    dr = ac + rng_r * np.array([np.cos(asp), np.sin(asp)])
    dv = (ac - dr) / np.linalg.norm(ac - dr) * v_dr
    acc = np.zeros(2)
    N = {"pn3": 3.0, "pn4": 4.0, "apn4": 4.0}.get(law, 4.0)
    lw = {"pn3": "pn", "pn4": "pn", "apn4": "apn"}.get(law, law)
    dmin, t, passed, first = np.inf, 0.0, False, False
    while t < t_max:
        a0, v0, acc0, d0 = ac.copy(), acv.copy(), acc.copy(), dr.copy()
        if ev is not None:
            if ev.react > 0:
                rel = dr - ac
                d = max(np.linalg.norm(rel), 1e-6)
                ev.alert(t, rel, -float(rel @ (dv - acv)) / d, nrm)
            y, yd, mult = ev.step(t, h, np.inf, pf.speed)
        else:
            y, yd, mult = 0.0, 0.0, 1.0
        s += pf.speed * mult * h
        p, tan, nrm, _ = pf.frame(s)
        ac = p + y * nrm
        acv = tan * pf.speed * mult + nrm * yd
        acc = (acv - v0) / h
        dv = guidance_step(lw, d0, dv, a0, v0, acc0, amax, h, N)
        dr = d0 + dv * h
        t += h
        miss = float(closest_approach(a0 - d0, ac - dr))
        dmin = min(dmin, miss)
        if miss <= contact:
            return True, t, dmin, not passed
        rel = ac - dr
        if not passed and np.linalg.norm(rel) < 100.0 and float(rel @ (acv - dv)) > 0:   # afastando: passou
            passed = True
    return False, np.nan, dmin, False


def _combo(args):
    law, pname, amax, a = args
    rng = np.random.default_rng(a.seed)                              # mesmos encontros para todas as leis
    prof, v_ac = CLASSES[pname] if a.realista else (PROFILES[pname], (22.0, 32.0))
    res = [encounter(law, prof, amax, a.contact, rng, t_max=a.t_max, v_ac=v_ac, v_dr=a.drone_speed)
           for _ in range(a.n)]
    hit = np.array([r[0] for r in res])
    first = np.array([r[3] for r in res])
    tt = np.array([r[1] for r in res])
    return {"lei": law, "evasao": pname, "amax": amax, "n": a.n,
            "p_contato": float(hit.mean()), "ic95": float(1.96 * np.sqrt(hit.mean() * (1 - hit.mean()) / a.n)),
            "p_1a": float(first.mean()), "ic95_1a": float(1.96 * np.sqrt(first.mean() * (1 - first.mean()) / a.n)),
            "t_med": float(np.nanmedian(tt)) if hit.any() else None,
            "dmin_med": float(np.median([r[2] for r in res]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--contact", type=float, default=2.0)
    ap.add_argument("--amax", type=float, nargs="+", default=[15.0, 30.0, 60.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=60.0, help="tempo maximo do encontro [s]")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--laws", nargs="+", default=list(LAWS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "guidance"))
    ap.add_argument("--realista", action="store_true",
                    help="classes das missoes realistas e interceptador de 65 m/s (contato 3 m)")
    ap.add_argument("--drone-speed", type=float, default=None, help="padrao: 42 (65 com --realista)")
    a = ap.parse_args()
    if a.drone_speed is None:
        a.drone_speed = 65.0 if a.realista else 42.0
    if a.realista and a.contact == 2.0:
        a.contact = 3.0
    out = os.path.join(a.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    combos = list(itertools.product(a.laws, CLASSES if a.realista else PROFILES, a.amax))
    with mp.Pool(a.workers) as pool:
        rows = pool.map(_combo, [(law, p, amax, a) for law, p, amax in combos], chunksize=1)
    for r in rows:
        print(f"{r['lei']:5s} {r['evasao']:12s} amax={r['amax']:5.1f}  P(contato)={100 * r['p_contato']:5.1f}% "
              f"1a passagem={100 * r['p_1a']:5.1f}%  t_med={r['t_med'] or float('nan'):5.1f}s", flush=True)
    with open(os.path.join(out, "guiagem.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "perfis": CLASSES if a.realista else PROFILES, "linhas": rows}, f, indent=2,
                  ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
