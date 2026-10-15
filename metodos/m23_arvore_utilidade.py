"""M23 -- Arvore de comportamento com utilidade afim (Feitosa, Ricardo Jr e Santos, COBEM 2025), adaptada a
interceptadores descartaveis. Cada drone decide sozinho com o quadro de radar comum (desempate pela ordem da fila):
  Responder: ha ameaca livre e plausivel?  -> alvo = a mais proxima (menor tempo de interceptacao)
             U = c1 * t_viagem/100 + c2 * ameaca(alvo) + c3 * ameaca composta / N + c4;  U > 0 -> avancar
  Ocioso:    esperar na base / manter a pista atual.
O ramo de reforco (pedido de apoio contra alvos resistentes) nao se aplica: um contato basta.

Treinavel: os coeficientes (c1..c4) sao ajustados por algoritmo genetico nos cenarios de treino
(``python scripts/treinar.py M23``) e gravados em modelos/M23_arvore_comportamento/coef.json. Sem esse arquivo, usa
os coeficientes padrao BT_COEF."""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv

from .base import Metodo, ler_json
from .comum import BIG, candidates, custo, priority_weight, rule_classify, slot_of, time_to_zone

BT_COEF = (-3.0, 2.0, 0.5, 0.5)      # (c1, c2, c3, c4) padrao, antes do ajuste
COEF_LO = (-10.0, -5.0, -5.0, -5.0)  # faixa de busca do GA
COEF_HI = (5.0, 10.0, 10.0, 5.0)


def bt_threat(env: RefuelEnv, p: dict) -> float:
    """Ameaca de uma pista: peso da zona com urgencia (decai com o tempo ate a zona, em escala de 2 min)."""
    tz = time_to_zone(env, p)
    return priority_weight(env, p) * (np.exp(-tz / 120.0) if np.isfinite(tz) else 0.1) \
        / env.rf.crash_penalty_priority


def behavior_tree(env: RefuelEnv, coef=BT_COEF, filt: bool = True) -> int:
    d = env.current_drone
    K = env.cfg.max_tracks
    if env.airborne(d) and not env.dr_lost[d]:
        return K                                                # ja engajado: mantem
    tr = candidates(env, env.free_tracks(d), filt)
    if not tr:
        return K
    cost = custo(env, [d], tr)[0]
    j = int(np.argmin(cost))
    if cost[j] >= BIG:
        return K
    comp = float(sum(bt_threat(env, p) for p in candidates(env, env._pic, filt)))
    c1, c2, c3, c4 = coef
    u = c1 * cost[j] / 100.0 + c2 * bt_threat(env, tr[j]) + c3 * comp / max(env.n_ac, 1) + c4
    return slot_of(env, tr[j]["id"]) if u > 0 else K


def bt_policy(coef=BT_COEF):
    coef = tuple(float(c) for c in coef)

    def pol(env: RefuelEnv) -> int:
        return behavior_tree(env, coef)
    pol.classify = rule_classify
    pol.coef = coef
    pol.__name__ = "behavior_tree"
    return pol


def carregar(arquivo: str | None):
    """Politica com os coeficientes de ``arquivo`` (coef.json) ou os padrao."""
    return bt_policy(ler_json(arquivo)["coef"] if arquivo else BT_COEF)


METODO = Metodo("M23", "arvore_comportamento", "Árvore (utilidade)", "Arquiteturas comportamentais", carregar,
                treino="coef")
