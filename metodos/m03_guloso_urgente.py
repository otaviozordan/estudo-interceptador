"""M3 -- Guloso (urgente): pista com a menor folga t^G_j - c_ij (tempo ate o solo estimado pelo radar menos o tempo
de interceptacao) entre as alcancaveis; sem nenhuma descendo, cai no M2."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, candidates, custo, slot_of, time_to_ground
from .m02_guloso_proximo import greedy_nearest


def greedy_urgent(env: RefuelEnv, filt: bool = True) -> int:
    tr = candidates(env, env.free_tracks(env.current_drone), filt)
    cost = custo(env, [env.current_drone], tr)[0]
    best, key = None, np.inf
    for p, c in zip(tr, cost):
        k = time_to_ground(env, p) - c
        if c < BIG and k < key:
            best, key = p, k
    return greedy_nearest(env, filt) if best is None else slot_of(env, best["id"])


METODO = Metodo("M03", "guloso_urgente", "Guloso (urgente)", "Regras simples", lambda _: politica(greedy_urgent))
