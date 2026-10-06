"""Compara tecnicas de alocacao drone -> aeronave para reabastecimento em voo, sobre os cenarios de missions/.

    python run_refuel.py                              # treina IL e PPO e compara com as linhas de base
    python run_refuel.py --missions missions --ppo-updates 200 --il-iters 8 --eval-episodes 10

Cada missao (construtor: aeronaves + radares + bases de drones + zonas prioritarias) e um cenario. Missoes sem bases
ou sem radares sao ignoradas. Os cenarios sao divididos em treino/teste (--test-frac) para medir generalizacao.
Os drones so usam pistas de radar (a maioria pode ser falso alarme: cada metodo decide se a pista e legitima);
cada drone reabastece uma aeronave; drones em voo podem trocar de alvo; queda de aeronave nao reabastecida e
penalizada (forte em zona prioritaria, branda fora). Metodos: aleatorio, gulosos, hungaro (com/sem filtro de
legitimidade) e redes neurais: imitacao do hungaro (DAgger), PPO do zero e imitacao + PPO.

Cada execucao grava em results/refuel/<AAAAMMDD_HHMMSS>/: config.json, log.txt, metricas.txt (tabelas),
comparacao.json (metricas por metodo e cenario, incl. precisao/recall legitima x falsa), pesos *.pt e gifs/ com
um episodio do melhor heuristico e da melhor rede em cada cenario de avaliacao (+ resumo do episodio em JSON).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from datetime import datetime

import matplotlib
import numpy as np
import torch

from swarm_sim.refuel.baselines import BASELINES
from swarm_sim.refuel.env import RefuelConfig, RefuelEnv, load_scenarios
from swarm_sim.refuel.learn import AllocNet, train_imitation, train_ppo
from swarm_sim.refuel.parallel import EpisodePool, make_policy

NETS = ("NN_imitacao", "PPO", "IL+PPO")


class Log:
    """print para a tela e para log.txt."""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def table(title, specs, scs, pool, episodes, res, log):
    log(f"\n== {title}: {len(scs)} cenario(s) x {episodes} episodios (sementes identicas para todos os metodos)")
    log(f"{'metodo':24s} {'sucesso':>8s} {'reabast.':>9s} {'caidas':>7s} {'c/prior.':>9s} {'penal.':>7s} "
        f"{'t_reab[s]':>10s} {'trocas':>7s} {'retorno':>8s} {'legit P/R':>11s}")
    for k, spec in specs.items():
        r = pool.evaluate(spec, scs, episodes)
        res.setdefault(title, {})[k] = r
        lg = r.get("legit")
        lg_s = f"{100 * lg['precisao']:4.0f}/{100 * lg['recall']:3.0f}%" if lg else "-"
        log(f"{k:24s} {100 * r['success']:7.0f}% {100 * r['rate']:8.1f}% {r['crashed']:7.2f} "
            f"{r['crashed_priority']:9.2f} {r['penalty']:7.2f} {r['mean_t']:10.1f} {r['retasks']:7.1f} "
            f"{r['return']:8.2f} {lg_s:>11s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", default=["missions"], help="pastas/arquivos/globs de missoes")
    ap.add_argument("--synthetic", action="store_true", help="DEM fractal em vez do GeoTIFF")
    ap.add_argument("--test-frac", type=float, default=0.25, help="fracao dos cenarios reservada ao teste")
    ap.add_argument("--il-iters", type=int, default=5)
    ap.add_argument("--ppo-updates", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=5, help="episodios por cenario na avaliacao")
    ap.add_argument("--gif-scenarios", type=int, default=3, help="max. de cenarios de avaliacao com GIF (0 = sem GIF)")
    ap.add_argument("--gif-seed", type=int, default=10_000, help="semente do episodio gravado")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processos para rodar episodios em paralelo")
    ap.add_argument("--out", default="results/refuel")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(a.out, stamp)
    os.makedirs(out, exist_ok=True)
    log = Log(os.path.join(out, "log.txt"))
    t0 = time.time()

    scs = load_scenarios(a.missions, a.synthetic)
    if not scs:
        raise SystemExit("nenhum cenario valido (precisa de aeronaves, radares e bases de drones)")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(scs))
    n_test = int(round(a.test_frac * len(scs))) if len(scs) >= 4 else 0
    test = [scs[i] for i in order[:n_test]]
    train = [scs[i] for i in order[n_test:]]
    log(f"execucao {stamp} -> {out}")
    log(f"{len(scs)} cenarios: {len(train)} treino ({', '.join(s.name for s in train)}) | "
        f"{len(test)} teste ({', '.join(s.name for s in test) or '-'})")
    cfg = RefuelConfig()
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"carimbo": stamp, "args": vars(a), "refuel_config": dataclasses.asdict(cfg),
                   "treino": [s.name for s in train], "teste": [s.name for s in test]}, f, indent=2, ensure_ascii=False)

    log(f"{a.workers} processo(s) para os episodios")
    with EpisodePool(scs, cfg, a.workers) as pool:
        log("== imitacao do hungaro com prioridade (DAgger)")
        il = train_imitation(train, cfg, iters=a.il_iters, log=log, pool=pool)
        log("== PPO do zero")
        ppo = train_ppo(train, cfg, updates=a.ppo_updates, seed=1, log=log, pool=pool)
        log("== imitacao + PPO")
        warm = AllocNet()
        warm.load_state_dict(il.state_dict())
        ilppo = train_ppo(train, cfg, warm, updates=a.ppo_updates, seed=2, lr=1e-4, log=log, pool=pool)
        for name, net in (("il", il), ("ppo", ppo), ("il_ppo", ilppo)):
            torch.save(net.state_dict(), os.path.join(out, f"{name}.pt"))
        specs = {k: ("base", k) for k in BASELINES}
        specs |= {k: ("net", net.state_dict(), True) for k, net in zip(NETS, (il, ppo, ilppo))}
        res: dict = {}
        table("treino", specs, train, pool, a.eval_episodes, res, log)
        if test:
            table("teste (cenarios nao vistos)", specs, test, pool, a.eval_episodes, res, log)

    # ---------------------------------------------------------------- GIFs: melhor heuristico e melhor rede
    ev_name = "teste (cenarios nao vistos)" if test else "treino"
    ev = res[ev_name]
    best_h = max((k for k in BASELINES if k != "aleatorio"), key=lambda k: ev[k]["return"])
    best_n = max(NETS, key=lambda k: ev[k]["return"])
    res["gifs"] = {"avaliacao": ev_name, "heuristico": best_h, "rede": best_n, "semente": a.gif_seed, "episodios": []}
    if a.gif_scenarios > 0:
        matplotlib.use("Agg")
        from swarm_sim.refuel.replay import record, save_gif
        log(f"\n== GIFs ({ev_name}): heuristico {best_h} x rede {best_n}, semente {a.gif_seed}")
        for sc in (test or train)[: a.gif_scenarios]:
            for kind, k in (("heuristico", best_h), ("rede", best_n)):
                env = RefuelEnv(sc, cfg, seed=a.gif_seed)
                policy = make_policy(specs[k])
                frames, ret = record(env, policy)
                s = env.summary() | {"return": ret, "metodo": k, "tipo": kind, "cenario": sc.name}
                tag = f"{kind}_{k}__{sc.name}".replace("/", "_").replace("+", "_")
                path = os.path.join(out, "gifs", tag + ".gif")
                save_gif(env, frames, path, f"{sc.name} | {kind}: {k} | seed {a.gif_seed} | {stamp}",
                         hasattr(policy, "classify"))
                with open(os.path.join(out, "gifs", tag + ".json"), "w", encoding="utf-8") as f:
                    json.dump(s, f, indent=2, ensure_ascii=False)
                res["gifs"]["episodios"].append(s | {"gif": os.path.relpath(path, out)})
                log(f"  {tag}.gif: {s['refueled']}/{s['n_aircraft']} reabastecidas, {s['crashed']} caidas, "
                    f"retorno {ret:.2f}")

    with open(os.path.join(out, "comparacao.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "log.txt"), encoding="utf-8") as f:
        txt = f.read()
    with open(os.path.join(out, "metricas.txt"), "w", encoding="utf-8") as f:
        f.write(txt[txt.index("\n== treino"):] if "\n== treino" in txt else txt)
    log(f"\nresultados em {out}/ (comparacao.json, metricas.txt, *.pt, gifs/) em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
