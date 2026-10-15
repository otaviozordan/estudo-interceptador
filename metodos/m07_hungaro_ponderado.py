"""M7 -- Hungaro ponderado: o problema de atribuicao com custo c_ij / w_j (beta = 1). Referencia dos testes
estatisticos e especialista imitado pelas redes (M24, M25)."""
from __future__ import annotations

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, politica
from .m06_hungaro import hungarian_lap


def hungarian_priority(env: RefuelEnv, filt: bool = True) -> int:
    return hungarian_lap(env, weighted=True, filt=filt)


METODO = Metodo("M07", "hungaro_prioridade", "Húngaro ponderado", "Atribuição ótima",
                lambda _: politica(hungarian_priority))
