"""M12 -- CBAA com comunicacao local: so drones a ate 3 km conversam, e ha 3 rodadas de consenso por decisao.
Drones de bases diferentes podem acreditar que ganharam a mesma pista (consciencia inconsistente); o simulador
resolve o conflito pela ordem de chegada."""
from __future__ import annotations

from .base import Metodo, politica_rodada
from .m11_cbaa import cbaa_round

ALCANCE = 3000.0         # [m]
RODADAS = 3


def cbaa_local(env):
    return cbaa_round(env, ALCANCE, RODADAS)


METODO = Metodo("M12", "cbaa_local", "CBAA com comunic. local", "Alocação distribuída",
                lambda _: politica_rodada(cbaa_local))
