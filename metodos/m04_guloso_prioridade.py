"""M4 -- Guloso (prioridade): min c_ij / w_j; pistas em zona prioritaria parecem dez vezes mais proximas."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, candidates, custo, priority_weight, slot_of


def greedy_priority(env: RefuelEnv, filt: bool = True) -> int:
    tr = candidates(env, env.free_tracks(env.current_drone), filt)
    cost = custo(env, [env.current_drone], tr)[0]
    if not tr or cost.min() >= BIG:
        return env.cfg.max_tracks
    w = np.array([priority_weight(env, p) for p in tr])
    sc = np.where(cost < BIG, cost / w, np.inf)
    return slot_of(env, tr[int(np.argmin(sc))]["id"])


METODO = Metodo("M04", "guloso_prioridade", "Guloso (prioridade)", "Regras simples",
                lambda _: politica(greedy_priority))
