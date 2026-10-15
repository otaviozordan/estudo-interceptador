"""M19 -- Recozimento simulado: parte do MMR, propoe realocacoes aleatorias de um drone e aceita pioras com
probabilidade exp(dF/T); 1500 movimentos, T0 = 0,2 |F0|, T <- 0,995 T."""
from __future__ import annotations

import numpy as np

from .base import Metodo, politica_rodada
from .comum import as_ids, objective, wta_setup
from .m13_mmr import mmr_assign


def sa(env, moves: int = 1500, t0: float = 0.2, alpha: float = 0.995):
    drones, tr, w = wta_setup(env)
    if w is None:
        return {d: None for d in drones}
    q, V, S, launch, count0 = w
    cap = env.cfg.max_per_track
    nd, nt = q.shape
    rng = np.random.default_rng(int(env.t * 1000) + 29)
    feas = [np.concatenate([[-1], np.nonzero(q[i] > 0)[0]]) for i in range(nd)]
    cur = mmr_assign(q, V, S, launch, count0, cap)
    f = objective(cur[None], q, V, S, launch, count0, cap)[0]
    best, fbest, T = cur.copy(), f, t0 * max(abs(f), 0.1)
    for _ in range(moves):
        a = cur.copy()
        i = int(rng.integers(nd))
        a[i] = rng.choice(feas[i])
        fa = objective(a[None], q, V, S, launch, count0, cap)[0]
        if fa >= f or rng.random() < np.exp((fa - f) / max(T, 1e-9)):
            cur, f = a, fa
            if f > fbest:
                best, fbest = cur.copy(), f
        T *= alpha
    return as_ids(drones, tr, best)


METODO = Metodo("M19", "sa", "Recozimento simulado", "Metaheurísticas", lambda _: politica_rodada(sa))
