"""M10 -- Guloso global: fixa o par drone-pista de maior lance w_j / c_ij (menor c_ij / w_j), remove o drone e a
pista e repete. E a forma centralizada do CBAA (Choi, Brunet e How, 2009) com comunicacao completa.
(A chave "cbaa" e historica: e o nome deste metodo nos resultados ja gravados.)"""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, pick, priority_weight, rodada


def greedy_global(env: RefuelEnv, filt: bool = True) -> int:
    drones, tr, cost, _ = rodada(env, filt)
    if not tr:
        return env.cfg.max_tracks
    c = np.where(cost < BIG, cost / np.array([priority_weight(env, p) for p in tr])[None, :], np.inf)
    assign: dict[int, int] = {}
    while np.isfinite(c).any():
        i, j = np.unravel_index(int(np.argmin(c)), c.shape)
        assign[int(i)] = int(j)
        c[i, :] = np.inf
        c[:, j] = np.inf
    return pick(env, drones, tr, assign)


METODO = Metodo("M10", "cbaa", "Guloso global", "Alocação distribuída", lambda _: politica(greedy_global))
