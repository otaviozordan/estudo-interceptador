"""M18 -- Colonia de formigas (Lee, Lee e Su, 2002): construcao probabilistica tau^alpha eta^beta, com "esperar"
como caminho de atratividade fixa 0,05; evaporacao rho e reforco da melhor solucao. 16 formigas, 15 iteracoes."""
from __future__ import annotations

import numpy as np

from .base import Metodo, politica_rodada
from .comum import as_ids, objective, score, wta_setup
from .m13_mmr import mmr_assign


def aco(env, ants: int = 16, iters: int = 15, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1):
    drones, tr, w = wta_setup(env)
    if w is None:
        return {d: None for d in drones}
    q, V, S, launch, count0 = w
    cap = env.cfg.max_per_track
    nd, nt = q.shape
    rng = np.random.default_rng(int(env.t * 1000) + 17)
    eta = np.concatenate([np.full((nd, 1), 0.05), np.maximum(score(q, V, S, launch), 1e-6) * (q > 0)], 1)
    tau = np.ones((nd, nt + 1))
    best = mmr_assign(q, V, S, launch, count0, cap)
    fbest = objective(best[None], q, V, S, launch, count0, cap)[0]
    for _ in range(iters):
        sols = np.full((ants, nd), -1)
        for a in range(ants):
            cnt = count0.copy()
            for i in rng.permutation(nd):
                p = tau[i] ** alpha * eta[i] ** beta
                p[1:][cnt >= cap] = 0.0
                if p.sum() <= 0:
                    continue
                j = int(rng.choice(nt + 1, p=p / p.sum())) - 1
                sols[a, i] = j
                if j >= 0:
                    cnt[j] += 1
        fs = objective(sols, q, V, S, launch, count0, cap)
        k = int(np.argmax(fs))
        if fs[k] > fbest:
            best, fbest = sols[k].copy(), fs[k]
        tau *= 1 - rho
        tau[np.arange(nd), best + 1] += rho * max(fbest, 1e-3)
    return as_ids(drones, tr, best)


METODO = Metodo("M18", "aco", "Colônia de formigas", "Metaheurísticas", lambda _: politica_rodada(aco))
