"""M22 -- Arvore de comportamento com alocacao por distancia (Ricardo Jr et al., 2023), adaptada: cada drone ocioso
engaja a ameaca livre e plausivel mais proxima se ela estiver a menos de R_e = 4 km; senao, espera. Em voo, mantem."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .comum import candidates, slot_of

ENGAGE_RANGE = 4000.0    # raio de engajamento [m]


def behavior_tree_distance(env: RefuelEnv, filt: bool = True) -> int:
    d = env.current_drone
    K = env.cfg.max_tracks
    if env.airborne(d) and not env.dr_lost[d]:
        return K
    tr = candidates(env, env.free_tracks(d), filt)
    if not tr:
        return K
    dist = np.array([np.hypot(p["n"] - env.dr_pos[d][0], p["e"] - env.dr_pos[d][1]) for p in tr])
    j = int(np.argmin(dist))
    return slot_of(env, tr[j]["id"]) if dist[j] <= ENGAGE_RANGE else K


METODO = Metodo("M22", "arvore_distancia", "Árvore (distância)", "Arquiteturas comportamentais",
                lambda _: politica(behavior_tree_distance))
