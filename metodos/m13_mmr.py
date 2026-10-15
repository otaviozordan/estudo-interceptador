"""M13 -- Maximo retorno marginal (den Broeder, Ellison e Emerling, 1959): designa um drone de cada vez ao par de
maior ganho V_j S~_j q_ij - l_i, atualiza a sobrevivencia da pista e para quando nenhum ganho e positivo (esperar).
Com ``cfg.max_per_track`` > 1 manda salvas. ``mmr_assign`` e a semente de M16--M20."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import pick, wta_setup


def mmr_assign(q, V, S, launch, count0, cap) -> np.ndarray:
    nd, nt = q.shape
    S = S.copy()
    cnt = count0.copy()
    out = np.full(nd, -1)
    free = np.ones(nd, bool)
    while free.any():
        gain = V[None, :] * S[None, :] * q - launch[:, None]
        gain[~free] = -np.inf
        gain[:, cnt >= cap] = -np.inf
        i, j = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if not np.isfinite(gain[i, j]) or gain[i, j] <= 0:
            break
        out[i], free[i] = j, False
        S[j] *= 1 - q[i, j]
        cnt[j] += 1
    return out


def mmr(env: RefuelEnv, filt: bool = True) -> int:
    drones, tr, w = wta_setup(env, filt)
    if w is None:
        return env.cfg.max_tracks
    a = mmr_assign(*w, env.cfg.max_per_track)
    return pick(env, drones, tr, {i: int(j) for i, j in enumerate(a)})


METODO = Metodo("M13", "mmr", "MMR", "WTA", lambda _: politica(mmr))
