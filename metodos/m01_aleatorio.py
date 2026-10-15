"""M1 -- Aleatorio: piso de desempenho. Lanca drones para pistas sorteadas, sem filtro, e nunca troca de alvo."""
from __future__ import annotations

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo


def random_policy(env: RefuelEnv) -> int:
    if env.airborne(env.current_drone):
        return env.cfg.max_tracks
    ok = [i for i, s in enumerate(env.slots) if s is not None]
    return int(env.rng.choice(ok)) if ok else env.cfg.max_tracks


METODO = Metodo("M01", "aleatorio", "Aleatório", "Regras simples", lambda _: random_policy)
