"""M14 -- WTA exato com no maximo um drone por pista, pelo hungaro: com n_bar = 1 o objetivo (eq. F) fica linear,
V_j S_j q_ij - l_i. Colunas replicadas pela capacidade restante de cada pista + uma coluna "esperar" por drone."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import Metodo, politica_rodada
from .comum import as_ids, score, wta_setup


def wta11_problem(env):
    """(drones, pistas, matriz de valor sc (-inf inviavel), capacidade restante) ou None se nao ha pistas."""
    drones, tr, w = wta_setup(env)
    if w is None:
        return drones, tr, None, None
    q, V, S, launch, count0 = w
    sc = np.where(q > 0, score(q, V, S, launch), -np.inf)
    cap = np.maximum(env.cfg.max_per_track - count0, 0)
    return drones, tr, sc, cap


def hungaro_wta(env):
    drones, tr, sc, cap = wta11_problem(env)
    if sc is None:
        return {d: None for d in drones}
    nd, nt = sc.shape
    cols = [j for j in range(nt) for _ in range(int(cap[j]))]
    C = np.zeros((nd, len(cols) + nd))
    if cols:
        C[:, :len(cols)] = np.where(np.isfinite(sc[:, cols]), -sc[:, cols], 1e9)
    r, c = linear_sum_assignment(C)
    a = np.full(nd, -1)
    for i, k in zip(r, c):
        if k < len(cols) and C[i, k] < 0:
            a[i] = cols[k]
    return as_ids(drones, tr, a)


METODO = Metodo("M14", "hungaro_wta", "Húngaro sobre WTA", "WTA", lambda _: politica_rodada(hungaro_wta))
