"""M20 -- Enxame de particulas com codificacao por chaves aleatorias: cada particula e uma tabela em [0, 1] (drone x
opcao, opcao 0 = esperar) decodificada pela opcao viavel de maior valor; uma particula comeca no MMR.
20 particulas, 25 iteracoes, w = 0,7, c1 = c2 = 1,5."""
from __future__ import annotations

import numpy as np

from .base import Metodo, politica_rodada
from .comum import as_ids, objective, wta_setup
from .m13_mmr import mmr_assign


def pso(env, particles: int = 20, iters: int = 25, w_in: float = 0.7, c1: float = 1.5, c2: float = 1.5):
    drones, tr, w = wta_setup(env)
    if w is None:
        return {d: None for d in drones}
    q, V, S, launch, count0 = w
    cap = env.cfg.max_per_track
    nd, nt = q.shape
    rng = np.random.default_rng(int(env.t * 1000) + 41)
    mask = np.concatenate([np.ones((nd, 1), bool), q > 0], 1)          # opcao 0 = esperar

    def decode(X):
        return np.argmax(np.where(mask[None], X, -np.inf), axis=2) - 1

    X = rng.random((particles, nd, nt + 1))
    m0 = mmr_assign(q, V, S, launch, count0, cap)
    X[0] = 0.0
    X[0, np.arange(nd), m0 + 1] = 1.0                                  # particula semeada com o MMR
    Vv = np.zeros_like(X)
    pb, fpb = X.copy(), objective(decode(X), q, V, S, launch, count0, cap)
    g = pb[int(np.argmax(fpb))].copy()
    fg = fpb.max()
    for _ in range(iters):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        Vv = w_in * Vv + c1 * r1 * (pb - X) + c2 * r2 * (g[None] - X)
        X = np.clip(X + Vv, 0.0, 1.0)
        f = objective(decode(X), q, V, S, launch, count0, cap)
        up = f > fpb
        pb[up], fpb[up] = X[up], f[up]
        if fpb.max() > fg:
            g, fg = pb[int(np.argmax(fpb))].copy(), fpb.max()
    return as_ids(drones, tr, decode(g[None])[0])


METODO = Metodo("M20", "pso", "Enxame de partículas", "Metaheurísticas", lambda _: politica_rodada(pso))
