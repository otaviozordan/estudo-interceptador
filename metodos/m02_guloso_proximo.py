"""M2 -- Guloso (proximo): o drone da vez vai a pista livre e plausivel que alcanca mais cedo (min c_ij)."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import BIG, candidates, custo, slot_of


def greedy_nearest(env: RefuelEnv, filt: bool = True) -> int:
    tr = candidates(env, env.free_tracks(env.current_drone), filt)
    cost = custo(env, [env.current_drone], tr)[0]
    if not tr or cost.min() >= BIG:
        return env.cfg.max_tracks
    return slot_of(env, tr[int(np.argmin(cost))]["id"])


METODO = Metodo("M02", "guloso_proximo", "Guloso (próximo)", "Regras simples", lambda _: politica(greedy_nearest))
