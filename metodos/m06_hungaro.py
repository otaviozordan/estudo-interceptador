"""M6 -- Hungaro (Kuhn, 1955; Munkres, 1957): atribuicao otima da rodada (drones da fila, na base ou em voo, x pistas
livres) minimizando a soma dos tempos de interceptacao (beta = 0). ``hungarian_lap`` serve tambem M7 e M8."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, candidates, custo, priority_weight, slot_of


def hungarian_lap(env: RefuelEnv, weighted: bool, filt: bool = True) -> int:
    """Problema de atribuicao (eq. lap); ``weighted`` divide o tempo pelo peso da zona (beta = 1)."""
    drones = list(env.queue)                                    # drones que ainda decidem nesta rodada
    busy = {int(env.dr_track[d]) for d in range(env.n_dr) if env.airborne(d) and d not in drones}
    tr = candidates(env, [p for p in env._pic if p["id"] not in busy], filt)
    if not tr:
        return env.cfg.max_tracks
    cost = custo(env, drones, tr)
    if weighted:
        cost = np.where(cost < BIG, cost / np.array([priority_weight(env, p) for p in tr])[None, :], BIG)
    rows, cols = linear_sum_assignment(cost)
    for r, k in zip(rows, cols):
        if drones[r] == env.current_drone and cost[r, k] < BIG:
            return slot_of(env, tr[k]["id"])
    return env.cfg.max_tracks


def hungarian(env: RefuelEnv, filt: bool = True) -> int:
    return hungarian_lap(env, weighted=False, filt=filt)


METODO = Metodo("M06", "hungaro", "Húngaro", "Atribuição ótima", lambda _: politica(hungarian))
