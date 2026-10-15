"""Descritor de um metodo de alocacao e os dois formatos de politica.

Interface comum (Secao 4 do artigo): ``policy(env) -> slot`` para o drone atual (``env.current_drone``), em que
``slot`` indexa ``env.slots`` e ``env.cfg.max_tracks`` e a acao nula (esperar na base / manter o alvo). A politica
pode ter o atributo ``classify(env, pistas) -> P(legitima)``, usado so para medir a decisao legitima/falso alarme.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from swarm_sim.refuel.env import RefuelEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(ROOT, "modelos")            # parametros treinados, uma pasta por metodo


@dataclass(frozen=True)
class Metodo:
    id: str                                        # "M07"
    chave: str                                     # nome nos resultados (episodes.jsonl): "hungaro_prioridade"
    nome: str                                      # "Húngaro ponderado"
    familia: str
    criar: Callable                                # criar(arquivo_do_modelo | None) -> policy
    treino: str | None = None                      # None | "coef" (arvore, GA) | "rede" (DAgger / PPO)
    rede: str | None = None                        # "deepsets" | "attn" (so treino == "rede")

    @property
    def pasta(self) -> str:
        """Pasta do metodo em modelos/: M24_IL-DS etc."""
        return f"{self.id}_{self.chave}"

    def arquivo(self, semente: int = 0, modelos: str = MODELOS) -> str | None:
        if self.treino == "coef":
            return os.path.join(modelos, self.pasta, "coef.json")
        if self.treino == "rede":
            return os.path.join(modelos, self.pasta, f"s{semente}.pt")
        return None

    def treinado(self, semente: int = 0, modelos: str = MODELOS) -> bool:
        f = self.arquivo(semente, modelos)
        return f is None or os.path.exists(f)

    def politica(self, semente: int = 0, modelos: str = MODELOS, arquivo: str | None = None):
        """Politica pronta para uso. Metodos treinaveis carregam o modelo de modelos/<pasta>/ (ou ``arquivo``)."""
        if self.treino is None:
            return self.criar(None)
        f = arquivo or self.arquivo(semente, modelos)
        if self.treino == "rede" and not os.path.exists(f):
            raise FileNotFoundError(f"{self.id} ({self.nome}) nao treinado: falta {f}\n"
                                    f"  treine com: python scripts/treinar.py {self.id} --semente {semente}")
        return self.criar(f if os.path.exists(f) else None)


def politica(fn, filt: bool = True):
    """Politica a partir de ``fn(env, filt) -> slot`` (metodos que decidem drone a drone ou refazem a rodada)."""
    def pol(env: RefuelEnv) -> int:
        return fn(env, filt)
    if filt:
        from .comum import rule_classify
        pol.classify = rule_classify
    pol.__name__ = getattr(fn, "__name__", "politica")
    return pol


def politica_rodada(resolver):
    """Politica a partir de ``resolver(env) -> {drone: id da pista ou None}``, que resolve a RODADA inteira (drones
    da fila x pistas). A solucao e guardada e reaproveitada pelos drones seguintes da mesma rodada."""
    from .comum import rule_classify, slot_of

    def pol(env: RefuelEnv) -> int:
        key = (env.t, env.seed)
        c = getattr(env, "_rc", None)
        if c is None or c[0] != key or c[1] is not resolver or env.current_drone not in c[2]:
            env._rc = (key, resolver, resolver(env))
        tid = env._rc[2].get(env.current_drone)
        return env.cfg.max_tracks if tid is None else slot_of(env, tid)
    pol.classify = rule_classify
    pol.__name__ = resolver.__name__
    return pol


def ler_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
