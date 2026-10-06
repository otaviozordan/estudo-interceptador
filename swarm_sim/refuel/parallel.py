"""Episodios em paralelo (processos) para treino e avaliacao: o rastreador de radar domina o custo de cada
episodio e os episodios sao independentes.

Tarefas (tuplas picklaveis): ("eval", i_cenario, semente, spec), ("dagger", i, semente, spec, especialista),
("ppo", i, semente, state_dict). spec = ("base", nome_da_linha_de_base) ou ("net", state_dict, greedy).
"""
from __future__ import annotations

import multiprocessing as mp

from .env import RefuelConfig, RefuelEnv, Scenario

_W: dict = {}


def _init(scenarios: list[Scenario], cfg: RefuelConfig):
    import torch
    torch.set_num_threads(1)
    _W["sc"], _W["cfg"] = scenarios, cfg


def make_policy(spec):
    from .baselines import BASELINES
    if spec[0] == "base":
        return BASELINES[spec[1]]
    from .learn import AllocNet, net_policy
    net = AllocNet()
    net.load_state_dict(spec[1])
    return net_policy(net, greedy=spec[2])


def _task(t):
    from .baselines import BASELINES, run_episode
    from .learn import AllocNet, dagger_episode, ppo_episode
    kind, i, seed = t[:3]
    sc, cfg = _W["sc"][i], _W["cfg"]
    if kind == "eval":
        return run_episode(make_policy(t[3]), sc, cfg, seed)
    env = RefuelEnv(sc, cfg, seed=seed)
    if kind == "dagger":
        return dagger_episode(env, make_policy(t[3]), BASELINES[t[4]])
    net = AllocNet()
    net.load_state_dict(t[3])
    net.eval()
    return ppo_episode(env, net)


class EpisodePool:
    """Pool de processos com os cenarios carregados uma vez por processo (workers <= 1: tudo no processo atual)."""

    def __init__(self, scenarios: list[Scenario], cfg: RefuelConfig, workers: int):
        self.scenarios = list(scenarios)
        self.pool = mp.get_context("spawn").Pool(workers, _init, (self.scenarios, cfg)) if workers > 1 else None
        if self.pool is None:
            _init(self.scenarios, cfg)

    def index(self, sc: Scenario) -> int:
        return next(i for i, s in enumerate(self.scenarios) if s is sc)

    def map(self, tasks: list) -> list:
        return self.pool.map(_task, tasks, chunksize=1) if self.pool else [_task(t) for t in tasks]

    def evaluate(self, spec, scenarios: list[Scenario], episodes_per_scenario: int | None = None,
                 seed0: int = 10_000) -> dict:
        """Igual a baselines.evaluate, com os episodios em paralelo (None = refuel.runs de cada missao)."""
        from .baselines import aggregate
        return aggregate(self.map([("eval", self.index(sc), seed0 + k, spec) for sc in scenarios
                                   for k in range(episodes_per_scenario or max(sc.m.refuel.runs, 1))]))

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
