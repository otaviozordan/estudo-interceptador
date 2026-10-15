"""M11 -- CBAA (Choi, Brunet e How, 2009): lances locais b_ij = w_j / max(c_ij, 1) e consenso de maximo entre
vizinhos do grafo de comunicacao. Aqui com grafo completo (estacao central) e ate 100 rodadas de consenso."""
from __future__ import annotations

import numpy as np

from .base import Metodo, politica_rodada
from .comum import BIG, as_ids, priority_weight, rodada

CBAA_RANGE = np.inf      # alcance de comunicacao entre drones [m] (inf = via estacao central, grafo completo)
CBAA_ROUNDS = 100        # rodadas de lance+consenso permitidas por rodada de decisao


def cbaa_solve(c: np.ndarray, adj: np.ndarray, max_rounds: int):
    """CBAA: c (agentes x tarefas) pontuacao (maior melhor, -inf inviavel); adj (agentes x agentes) grafo.
    Cada agente guarda x (sua tarefa), y (lances vencedores conhecidos) e z (vencedores). Devolve (x, rodadas)."""
    n, m = c.shape
    x = np.full(n, -1)
    y = np.full((n, m), -np.inf)
    z = np.full((n, m), -1)
    A = adj | np.eye(n, dtype=bool)
    for rnd in range(1, max_rounds + 1):
        changed = False
        for i in range(n):                                          # fase de lances
            if x[i] < 0:
                h = np.isfinite(c[i]) & ((c[i] > y[i]) | ((c[i] == y[i]) & ((z[i] < 0) | (i < z[i]))))
                if h.any():
                    j = int(np.argmax(np.where(h, c[i], -np.inf)))
                    x[i], y[i, j], z[i, j] = j, c[i, j], i
                    changed = True
        y0, z0 = y.copy(), z.copy()
        for i in range(n):                                          # consenso (max-consenso sincrono)
            nb = np.nonzero(A[i])[0]
            yy = y0[nb]
            zz = np.where(z0[nb] < 0, n + 1, z0[nb])
            key = np.where(np.isfinite(yy), yy, -1e300) - 1e-12 * zz   # desempate: menor indice vence
            k = nb[np.argmax(key, axis=0)]
            y[i] = y0[k, np.arange(m)]
            z[i] = z0[k, np.arange(m)]
            if x[i] >= 0 and z[i, x[i]] != i:                       # perdeu a tarefa: libera
                x[i] = -1
                changed = True
        if not changed:
            return x, rnd
    return x, max_rounds


def cbaa_round(env, rng_m: float, rounds: int):
    drones, tr, cost, _ = rodada(env)
    if not tr:
        return {d: None for d in drones}
    w = np.array([priority_weight(env, p) for p in tr])
    c = np.where(cost < BIG, w[None, :] / np.maximum(cost, 1.0), -np.inf)
    P = env.dr_pos[drones]
    adj = np.linalg.norm(P[:, None] - P[None], axis=2) <= rng_m
    x, _ = cbaa_solve(c, adj, rounds)
    return as_ids(drones, tr, x)


def cbaa_dist(env):
    return cbaa_round(env, CBAA_RANGE, CBAA_ROUNDS)


METODO = Metodo("M11", "cbaa_dist", "CBAA", "Alocação distribuída", lambda _: politica_rodada(cbaa_dist))
