"""M16 -- Busca local (VLSN simplificada, Ahuja et al., 2007): parte do MMR e aplica a melhor realocacao de um drone
ou troca de alvos entre dois drones enquanto houver ganho no objetivo (eq. F). Ate 30 iteracoes."""
from __future__ import annotations

import numpy as np

from .base import Metodo, politica_rodada
from .comum import as_ids, objective, wta_setup
from .m13_mmr import mmr_assign


def vlsn(env, max_iter: int = 30):
    drones, tr, w = wta_setup(env)
    if w is None:
        return {d: None for d in drones}
    q, V, S, launch, count0 = w
    cap = env.cfg.max_per_track
    nd, nt = q.shape
    cur = mmr_assign(q, V, S, launch, count0, cap)
    f = objective(cur[None], q, V, S, launch, count0, cap)[0]
    feas = [np.concatenate([[-1], np.nonzero(q[i] > 0)[0]]) for i in range(nd)]
    for _ in range(max_iter):
        cand = []
        for i in range(nd):                                          # realocacao de um drone
            for j in feas[i]:
                if j != cur[i]:
                    a = cur.copy()
                    a[i] = j
                    cand.append(a)
        for i in range(nd):                                          # troca de alvos entre dois drones
            for k in range(i + 1, nd):
                if cur[i] != cur[k] and (cur[k] < 0 or q[i, cur[k]] > 0) and (cur[i] < 0 or q[k, cur[i]] > 0):
                    a = cur.copy()
                    a[i], a[k] = cur[k], cur[i]
                    cand.append(a)
        if not cand:
            break
        C = np.array(cand)
        fc = objective(C, q, V, S, launch, count0, cap)
        b = int(np.argmax(fc))
        if fc[b] <= f + 1e-9:
            break
        cur, f = C[b], fc[b]
    return as_ids(drones, tr, cur)


METODO = Metodo("M16", "vlsn", "Busca local (VLSN)", "WTA", lambda _: politica_rodada(vlsn))
