"""Compara tecnicas de alocacao drone -> aeronave para reabastecimento em voo, sobre os cenarios de missions/.

    python run_refuel.py                              # treina IL e PPO e compara com as linhas de base
    python run_refuel.py --missions missions --ppo-updates 200 --il-iters 8 --eval-episodes 10

Cada missao (construtor: aeronaves + radares + bases de drones + zonas prioritarias) e um cenario. Missoes sem bases
ou sem radares sao ignoradas. Os cenarios sao divididos em treino/teste (--test-frac) para medir generalizacao.
Os drones so usam pistas de radar; cada drone reabastece uma aeronave; queda de aeronave nao reabastecida e
penalizada (forte em zona prioritaria, branda fora). Metodos: aleatorio, gulosos, hungaro (com/sem prioridade) e
redes neurais: imitacao do hungaro (DAgger), PPO do zero e imitacao + PPO. Resultados em results/refuel/.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from swarm_sim.refuel.baselines import BASELINES, evaluate
from swarm_sim.refuel.env import RefuelConfig, load_scenarios
from swarm_sim.refuel.learn import AllocNet, net_policy, train_imitation, train_ppo


def table(title, pol, scs, cfg, episodes, res):
    print(f"\n== {title}: {len(scs)} cenario(s) x {episodes} episodios (sementes identicas para todos os metodos)")
    print(f"{'metodo':20s} {'sucesso':>8s} {'reabast.':>9s} {'caidas':>7s} {'c/prior.':>9s} {'penal.':>7s} {'t_reab[s]':>10s} {'retorno':>8s}")
    for k, p in pol.items():
        r = evaluate(p, scs, cfg, episodes)
        res.setdefault(title, {})[k] = r
        print(f"{k:20s} {100 * r['success']:7.0f}% {100 * r['rate']:8.1f}% {r['crashed']:7.2f} "
              f"{r['crashed_priority']:9.2f} {r['penalty']:7.2f} {r['mean_t']:10.1f} {r['return']:8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", default=["missions"], help="pastas/arquivos/globs de missoes")
    ap.add_argument("--synthetic", action="store_true", help="DEM fractal em vez do GeoTIFF")
    ap.add_argument("--test-frac", type=float, default=0.25, help="fracao dos cenarios reservada ao teste")
    ap.add_argument("--il-iters", type=int, default=5)
    ap.add_argument("--ppo-updates", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=5, help="episodios por cenario na avaliacao")
    ap.add_argument("--out", default="results/refuel")
    a = ap.parse_args()

    scs = load_scenarios(a.missions, a.synthetic)
    if not scs:
        raise SystemExit("nenhum cenario valido (precisa de aeronaves, radares e bases de drones)")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(scs))
    n_test = int(round(a.test_frac * len(scs))) if len(scs) >= 4 else 0
    test = [scs[i] for i in order[:n_test]]
    train = [scs[i] for i in order[n_test:]]
    print(f"{len(scs)} cenarios: {len(train)} treino ({', '.join(s.name for s in train)}) | "
          f"{len(test)} teste ({', '.join(s.name for s in test) or '-'})")
    cfg = RefuelConfig()
    os.makedirs(a.out, exist_ok=True)

    pol = dict(BASELINES)
    print("== imitacao do hungaro com prioridade (DAgger)")
    il = train_imitation(train, cfg, iters=a.il_iters)
    pol["NN_imitacao"] = net_policy(il)
    print("== PPO do zero")
    ppo = train_ppo(train, cfg, updates=a.ppo_updates, seed=1)
    pol["PPO"] = net_policy(ppo)
    print("== imitacao + PPO")
    warm = AllocNet()
    warm.load_state_dict(il.state_dict())
    ilppo = train_ppo(train, cfg, warm, updates=a.ppo_updates, seed=2, lr=1e-4)
    pol["IL+PPO"] = net_policy(ilppo)
    for name, net in (("il", il), ("ppo", ppo), ("il_ppo", ilppo)):
        torch.save(net.state_dict(), os.path.join(a.out, f"{name}.pt"))

    res: dict = {}
    table("treino", pol, train, cfg, a.eval_episodes, res)
    if test:
        table("teste (cenarios nao vistos)", pol, test, cfg, a.eval_episodes, res)
    with open(os.path.join(a.out, "comparacao.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nresultados em {a.out}/comparacao.json (por cenario) e pesos *.pt")


if __name__ == "__main__":
    main()
