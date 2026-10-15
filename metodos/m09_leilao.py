"""M9 -- Leilao de Bertsekas (1988) sobre os mesmos beneficios do M7 (a_ij = -c_ij / w_j)."""
from __future__ import annotations

import numpy as np

from .base import Metodo, politica_rodada
from .comum import BIG, as_ids, priority_weight, rodada


def auction(benefit: np.ndarray, eps: float | None = None):
    """Leilao direto (Gauss-Seidel) para o problema assimetrico (n_pessoas <= n_objetos), com precos iniciais nulos e
    uma unica fase de epsilon: assim os objetos sem dono mantem preco minimo (zero) e a solucao fica a menos de
    n*eps do otimo (o escalonamento de epsilon perderia essa garantia no caso assimetrico).
    benefit: -inf = par inviavel. Devolve (objeto de cada pessoa ou -1, numero de lances)."""
    n, m = benefit.shape
    fin = np.isfinite(benefit)
    if not fin.any():
        return np.full(n, -1), 0
    rng_b = float(np.ptp(benefit[fin])) or 1.0
    eps = eps or rng_b * 1e-3 / (n + 1)
    price = np.zeros(m)
    owner = np.full(m, -1)
    assign = np.full(n, -1)
    free = [i for i in range(n) if fin[i].any()]
    bids = 0
    while free:
        i = free.pop()
        v = np.where(fin[i], benefit[i] - price, -np.inf)
        j = int(np.argmax(v))
        best = v[j]
        v[j] = -np.inf
        second = float(v.max()) if np.isfinite(v).any() else best - rng_b
        price[j] += best - second + eps
        bids += 1
        if owner[j] >= 0:
            assign[owner[j]] = -1
            free.append(int(owner[j]))
        owner[j], assign[i] = i, j
    return assign, bids


def leilao(env):
    drones, tr, cost, _ = rodada(env)
    if not tr:
        return {d: None for d in drones}
    w = np.array([priority_weight(env, p) for p in tr])
    ben = np.where(cost < BIG, -cost / w[None, :], -np.inf)          # mesmo objetivo do hungaro ponderado
    if len(drones) <= len(tr):
        a, _ = auction(ben)
    else:                                                            # mais drones que pistas: pistas "compram"
        b, _ = auction(ben.T)
        a = np.full(len(drones), -1)
        for j, i in enumerate(b):
            if i >= 0:
                a[i] = j
    return as_ids(drones, tr, a)


METODO = Metodo("M09", "leilao", "Leilão (Bertsekas)", "Atribuição ótima", lambda _: politica_rodada(leilao))
