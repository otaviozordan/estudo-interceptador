"""M5 -- TEWA sequencial (Roux e van Vuuren, 2007): pistas ordenadas por ameaca theta_j = w_P / (1 + t^Z_j) (ou
w_O / 1000 se nao ruma a uma zona); cada uma recebe o drone livre da rodada que a intercepta mais cedo."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, pick, rodada, time_to_zone


def tewa(env: RefuelEnv, filt: bool = True) -> int:
    drones, tr, cost, _ = rodada(env, filt)
    if not tr:
        return env.cfg.max_tracks
    pri, oth = env.rf.crash_penalty_priority, env.rf.crash_penalty_other
    threat = []
    for p in tr:
        tz = time_to_zone(env, p)
        threat.append(pri / (1.0 + tz) if np.isfinite(tz) else oth / 1e3)   # sem rumo a zona: fila por ultimo
    free = set(range(len(drones)))
    assign: dict[int, int] = {}
    for j in np.argsort(-np.array(threat), kind="stable"):
        if not free:
            break
        cand = [i for i in free if cost[i, j] < BIG]
        if cand:
            i = min(cand, key=lambda i: cost[i, j])
            assign[i] = int(j)
            free.discard(i)
    return pick(env, drones, tr, assign)


METODO = Metodo("M05", "tewa", "TEWA sequencial", "Regras simples", lambda _: politica(tewa))
