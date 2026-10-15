"""M21 -- Hungaro com reserva (licao do WTA dinamico, Hosein e Athans, 1990): resolve como o M7, mas um drone da
base so e lancado contra pista de baixa prioridade se restarem mais de 20 % da frota ociosos depois do lancamento.
Pistas em zona prioritaria podem usar a reserva."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, custo, priority_weight, rodada, slot_of

RESERVE_FRAC = 0.2      # fracao dos drones mantida em reserva


def hungarian_reserve(env: RefuelEnv, filt: bool = True) -> int:
    drones, tr, cost, _ = rodada(env, filt)
    if not tr:
        return env.cfg.max_tracks
    tr = [p for p in tr if p["id"] not in env.targeted or any(  # 1 drone por pista (como o hungaro)
        env.airborne(d) and d in drones and env.dr_track[d] == p["id"] for d in range(env.n_dr))]
    if not tr:
        return env.cfg.max_tracks
    cost = custo(env, drones, tr)
    w = np.array([priority_weight(env, p) for p in tr])
    cost = np.where(cost < BIG, cost / w[None, :], BIG)
    idle = int(np.sum(env.dr_state == 0))
    reserve = int(round(RESERVE_FRAC * env.n_dr))
    low = w < env.rf.crash_penalty_priority
    rows, cols = linear_sum_assignment(cost)
    launches = 0
    for r, k in sorted(zip(rows, cols), key=lambda rk: cost[rk[0], rk[1]]):
        if cost[r, k] >= BIG:
            continue
        d = drones[r]
        if not env.airborne(d):
            if low[k] and idle - launches <= reserve:                  # reserva: so para pistas prioritarias
                continue
            launches += 1
        if d == env.current_drone:
            return slot_of(env, tr[k]["id"])
    return env.cfg.max_tracks


METODO = Metodo("M21", "hungaro_reserva", "Húngaro com reserva", "Alocação dinâmica",
                lambda _: politica(hungarian_reserve))
