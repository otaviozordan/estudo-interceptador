"""Os 29 metodos de alocacao do artigo (M1--M29), um modulo por metodo, todos na mesma interface:
``policy(env) -> slot`` (ver ``base.py``).

    from metodos import obter
    m = obter("M07")              # ou "7", "m7", "hungaro_prioridade"
    policy = m.politica()         # treinaveis (M23--M29) carregam modelos/<id>_<chave>/

Organizacao:
    base.py      descritor Metodo e os formatos de politica (por drone / por rodada)
    comum.py     elementos comuns E1--E4 (filtro, custo com histerese, peso de prioridade, modelo de sucesso)
    mNN_*.py     um metodo por arquivo (M1--M23)
    aprendidos.py, redes.py   M24--M29 (DeepSets / atencao)
    treino.py    ajuste dos treinaveis (GA da arvore, DAgger, PPO)
    paralelo.py  episodios em paralelo (avaliacao e treino)
"""
from __future__ import annotations

from . import (m01_aleatorio, m02_guloso_proximo, m03_guloso_urgente, m04_guloso_prioridade, m05_tewa, m06_hungaro,
               m07_hungaro_ponderado, m08_hungaro_sem_filtro, m09_leilao, m10_guloso_global, m11_cbaa, m12_cbaa_local,
               m13_mmr, m14_hungaro_wta, m15_milp, m16_vlsn, m17_ga, m18_aco, m19_sa, m20_pso, m21_hungaro_reserva,
               m22_arvore_distancia, m23_arvore_utilidade)
from .aprendidos import METODOS as _APRENDIDOS
from .base import MODELOS, Metodo

_MODULOS = [m01_aleatorio, m02_guloso_proximo, m03_guloso_urgente, m04_guloso_prioridade, m05_tewa, m06_hungaro,
            m07_hungaro_ponderado, m08_hungaro_sem_filtro, m09_leilao, m10_guloso_global, m11_cbaa, m12_cbaa_local,
            m13_mmr, m14_hungaro_wta, m15_milp, m16_vlsn, m17_ga, m18_aco, m19_sa, m20_pso, m21_hungaro_reserva,
            m22_arvore_distancia, m23_arvore_utilidade]

METODOS: dict[str, Metodo] = {m.METODO.id: m.METODO for m in _MODULOS} | {m.id: m for m in _APRENDIDOS}
POR_CHAVE: dict[str, Metodo] = {m.chave: m for m in METODOS.values()}

# politicas sem treino por chave (M23 com os coeficientes padrao), como nos resultados gravados
METHODS = {m.chave: m.criar(None) for m in METODOS.values() if m.treino != "rede"}


def obter(x: str) -> Metodo:
    """Metodo por id ("M07", "m7", "7") ou chave ("hungaro_prioridade", "IL-DS")."""
    s = str(x).strip()
    if s in POR_CHAVE:
        return POR_CHAVE[s]
    d = s[1:] if s[:1] in "mM" else s
    if d.isdigit() and f"M{int(d):02d}" in METODOS:
        return METODOS[f"M{int(d):02d}"]
    raise KeyError(f"metodo desconhecido: {x!r} (use M1..M29 ou uma chave; liste com: python rodar.py --listar)")


__all__ = ["METODOS", "POR_CHAVE", "METHODS", "MODELOS", "Metodo", "obter"]
