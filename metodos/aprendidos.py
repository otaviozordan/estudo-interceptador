"""M24--M29 -- Politicas aprendidas: uma rede compartilhada por todos os drones (DeepSets ou atencao), treinada por
imitacao do M7 (DAgger), por PPO a partir da imitacao, ou por PPO do zero. Os pesos ficam em
modelos/<id>_<chave>/s<semente>.pt (uma rede por semente de treino); treine com ``python scripts/treinar.py M24``.

A chave (IL-DS, IL+PPO-ATT, ...) e o nome do metodo nos resultados ja gravados (episodes.jsonl)."""
from __future__ import annotations

from .base import Metodo


def _carregar(kind: str):
    def criar(arquivo: str):
        from .redes import load_net, net_policy
        return net_policy(load_net(arquivo, kind), greedy=True)
    return criar


def _m(id_, chave, nome, kind):
    return Metodo(id_, chave, nome, "Políticas aprendidas", _carregar(kind), treino="rede", rede=kind)


METODOS = [
    _m("M24", "IL-DS", "Imitação (DeepSets)", "deepsets"),
    _m("M25", "IL-ATT", "Imitação (atenção)", "attn"),
    _m("M26", "IL+PPO-DS", "Imitação+PPO (DeepSets)", "deepsets"),
    _m("M27", "IL+PPO-ATT", "Imitação+PPO (atenção)", "attn"),
    _m("M28", "PPO-DS", "PPO (DeepSets)", "deepsets"),
    _m("M29", "PPO-ATT", "PPO (atenção)", "attn"),
]

# como cada um e treinado: ("il", None) imitacao | ("ppo", base) refino de outra rede | ("ppo", None) do zero
RECEITA = {"M24": ("il", None), "M25": ("il", None), "M26": ("ppo", "M24"), "M27": ("ppo", "M25"),
           "M28": ("ppo", None), "M29": ("ppo", None)}
