"""M15 -- O mesmo problema do M14 como programa linear inteiro, resolvido pelo HiGHS (ramificacao e corte), como na
parte de alocacao do MPC inteiro de Afonso et al. (2020) e Cruz et al. (2025). Limite de 5 s por rodada."""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .base import Metodo, politica_rodada
from .comum import as_ids
from .m14_hungaro_wta import wta11_problem

TIME_LIMIT = 5.0


def milp_wta(env):
    drones, tr, sc, cap = wta11_problem(env)
    if sc is None:
        return {d: None for d in drones}
    nd, nt = sc.shape
    I, J = np.nonzero(np.isfinite(sc) & (sc > 0))
    if not len(I):
        return {d: None for d in drones}
    n = len(I)
    A = np.zeros((nd + nt, n))
    A[I, np.arange(n)] = 1.0
    A[nd + J, np.arange(n)] = 1.0
    ub = np.concatenate([np.ones(nd), cap])
    res = milp(-sc[I, J], constraints=LinearConstraint(A, -np.inf, ub), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": TIME_LIMIT})
    a = np.full(nd, -1)
    if res.x is not None:
        for k in np.nonzero(res.x > 0.5)[0]:
            a[I[k]] = J[k]
    return as_ids(drones, tr, a)


METODO = Metodo("M15", "milp_wta", "MILP (HiGHS)", "WTA", lambda _: politica_rodada(milp_wta))
