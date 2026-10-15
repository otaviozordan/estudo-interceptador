"""M17 -- Algoritmo genetico (Lee, Su e Lee, 2003) sobre o objetivo (eq. F), semeado com o MMR ("eugenia gulosa"):
torneio de 3, cruzamento uniforme, mutacao 1/n, elitismo de 2; populacao 40, 40 geracoes."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import objective, pick, wta_setup
from .m13_mmr import mmr_assign


def ga(env: RefuelEnv, filt: bool = True, pop: int = 40, gens: int = 40) -> int:
    drones, tr, w = wta_setup(env, filt)
    if w is None:
        return env.cfg.max_tracks
    q, V, S, launch, count0 = w
    cap = env.cfg.max_per_track
    nd, nt = q.shape
    rng = np.random.default_rng(int(env.t * 1000) + env.current_drone)    # deterministico por estado
    feas = [np.concatenate([[-1], np.where(q[i] > 0)[0]]) for i in range(nd)]
    P = np.array([[rng.choice(feas[i]) for i in range(nd)] for _ in range(pop)])
    P[0] = mmr_assign(q, V, S, launch, count0, cap)                        # semente gulosa ("eugenia")
    P[1] = -1
    fit = objective(P, q, V, S, launch, count0, cap)
    pm = 1.0 / max(nd, 1)
    for _ in range(gens):
        a, b = rng.integers(pop, size=(2, pop, 3))
        pa = P[a[np.arange(pop), np.argmax(fit[a], 1)]]
        pb = P[b[np.arange(pop), np.argmax(fit[b], 1)]]
        mask = rng.random((pop, nd)) < 0.5
        child = np.where(mask, pa, pb)
        mut = rng.random((pop, nd)) < pm
        for r, i in zip(*np.nonzero(mut)):
            child[r, i] = rng.choice(feas[i])
        cf = objective(child, q, V, S, launch, count0, cap)
        elite = np.argsort(-fit)[:2]
        P = np.vstack([P[elite], child[np.argsort(-cf)[: pop - 2]]])
        fit = np.concatenate([fit[elite], np.sort(cf)[::-1][: pop - 2]])
    best = P[int(np.argmax(fit))]
    return pick(env, drones, tr, {i: int(j) for i, j in enumerate(best)})


METODO = Metodo("M17", "ga", "Algoritmo genético", "Metaheurísticas", lambda _: politica(ga))
