"""Validacao das implementacoes e custo computacional por rodada de decisao.

    python scripts/experimentos/check_equivalence.py [--n-scen 8] [--paper paper]

O episodio e conduzido pelo hungaro ponderado; em cada rodada de decisao (instante com fila nao vazia) TODOS os
solucionadores resolvem a mesma instancia (drones da fila x pistas). Mede-se:
* concordancia entre pares que a teoria diz equivalentes: leilao x hungaro ponderado, CBAA distribuido (grafo
  completo) x guloso global, MILP (HiGHS) x hungaro sobre o WTA 1-1;
* para os metodos do objetivo WTA (MMR, busca local, GA, ACO, SA, PSO, hungaro WTA): valor do objetivo relativo ao
  melhor encontrado na rodada;
* tempo de solucao por rodada (mediana e percentil 95) em funcao do tamanho da instancia.
Grava results/equivalencia.json e <paper>/tabelas/equivalencia.tex.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np                                                  # noqa: E402
from scipy.optimize import linear_sum_assignment                    # noqa: E402

from metodos import METHODS                                         # noqa: E402
from metodos.comum import BIG, as_ids, objective, priority_weight, rodada, wta_setup   # noqa: E402
from metodos.m09_leilao import leilao                               # noqa: E402
from metodos.m11_cbaa import cbaa_dist                              # noqa: E402
from metodos.m13_mmr import mmr_assign                              # noqa: E402
from metodos.m14_hungaro_wta import hungaro_wta                     # noqa: E402
from metodos.m15_milp import milp_wta                               # noqa: E402
from metodos.m16_vlsn import vlsn                                   # noqa: E402
from metodos.m18_aco import aco                                     # noqa: E402
from metodos.m19_sa import sa                                       # noqa: E402
from metodos.m20_pso import pso                                     # noqa: E402
from swarm_sim.refuel.env import RefuelConfig, RefuelEnv, Scenario


def hung_prio(env):
    drones, tr, cost, _ = rodada(env)
    if not tr:
        return {d: None for d in drones}
    c = np.where(cost < BIG, cost / np.array([priority_weight(env, p) for p in tr])[None, :], BIG)
    r, k = linear_sum_assignment(c)
    a = np.full(len(drones), -1)
    for i, j in zip(r, k):
        if c[i, j] < BIG:
            a[i] = j
    return as_ids(drones, tr, a)


def greedy_global(env):
    drones, tr, cost, _ = rodada(env)
    if not tr:
        return {d: None for d in drones}
    c = np.where(cost < BIG, cost / np.array([priority_weight(env, p) for p in tr])[None, :], np.inf)
    a = np.full(len(drones), -1)
    while np.isfinite(c).any():
        i, j = np.unravel_index(int(np.argmin(c)), c.shape)
        a[i] = j
        c[i, :] = np.inf
        c[:, j] = np.inf
    return as_ids(drones, tr, a)


def mmr_full(env):
    drones, tr, w = wta_setup(env)
    if w is None:
        return {d: None for d in drones}
    return as_ids(drones, tr, mmr_assign(*w, env.cfg.max_per_track))


def ga_full(env):
    """GA da rodada inteira (o metodo do benchmark recalcula por drone; aqui a solucao da rodada)."""
    drones, tr, w = wta_setup(env)
    if w is None:
        return {d: None for d in drones}
    q, V, S, launch, count0 = w
    cap = env.cfg.max_per_track
    nd, nt = q.shape
    rng = np.random.default_rng(int(env.t * 1000))
    feas = [np.concatenate([[-1], np.where(q[i] > 0)[0]]) for i in range(nd)]
    pop, gens = 40, 40
    P = np.array([[rng.choice(feas[i]) for i in range(nd)] for _ in range(pop)])
    P[0] = mmr_assign(q, V, S, launch, count0, cap)
    P[1] = -1
    fit = objective(P, q, V, S, launch, count0, cap)
    for _ in range(gens):
        a, b = rng.integers(pop, size=(2, pop, 3))
        pa = P[a[np.arange(pop), np.argmax(fit[a], 1)]]
        pb = P[b[np.arange(pop), np.argmax(fit[b], 1)]]
        child = np.where(rng.random((pop, nd)) < 0.5, pa, pb)
        for r, i in zip(*np.nonzero(rng.random((pop, nd)) < 1.0 / max(nd, 1))):
            child[r, i] = rng.choice(feas[i])
        cf = objective(child, q, V, S, launch, count0, cap)
        elite = np.argsort(-fit)[:2]
        P = np.vstack([P[elite], child[np.argsort(-cf)[: pop - 2]]])
        fit = np.concatenate([fit[elite], np.sort(cf)[::-1][: pop - 2]])
    return as_ids(drones, tr, P[int(np.argmax(fit))])


SOLVERS = {"hungaro_prioridade": hung_prio, "leilao": leilao, "guloso_global": greedy_global,
           "cbaa_dist": cbaa_dist, "hungaro_wta": hungaro_wta, "milp_wta": milp_wta, "mmr": mmr_full,
           "vlsn": vlsn, "ga": ga_full, "aco": aco, "sa": sa, "pso": pso}
PAIRS = [("leilao", "hungaro_prioridade"), ("cbaa_dist", "guloso_global"), ("milp_wta", "hungaro_wta")]
WTA_OBJ = ["hungaro_wta", "mmr", "vlsn", "ga", "aco", "sa", "pso"]
NAMES = {"hungaro_prioridade": "Húngaro ponderado", "leilao": "Leilão (Bertsekas)", "guloso_global": "Guloso global",
         "cbaa_dist": "CBAA distribuído", "hungaro_wta": "Húngaro sobre WTA", "milp_wta": "MILP (HiGHS)",
         "mmr": "MMR", "vlsn": "Busca local (VLSN)", "ga": "Algoritmo genético", "aco": "Colônia de formigas",
         "sa": "Recozimento simulado", "pso": "Enxame de partículas"}


def objective_of(env, sol: dict) -> float:
    drones, tr, w = wta_setup(env)
    if w is None:
        return 0.0
    idx = {p["id"]: j for j, p in enumerate(tr)}
    a = np.array([idx.get(sol.get(d), -1) if sol.get(d) is not None else -1 for d in drones])
    return float(objective(a[None], *w, env.cfg.max_per_track)[0])


def lap_cost(env, sol: dict) -> float:
    """Custo do problema (lap) com beta = 1 (soma de c_ij / w_j), com mais designacoes em primeiro lugar."""
    drones, tr, cost, _ = rodada(env)
    idx = {p["id"]: j for j, p in enumerate(tr)}
    w = np.array([priority_weight(env, p) for p in tr])
    tot, n = 0.0, 0
    for i, d in enumerate(drones):
        j = idx.get(sol.get(d)) if sol.get(d) is not None else None
        if j is not None and cost[i, j] < BIG:
            tot += cost[i, j] / w[j]
            n += 1
    return tot - 1e6 * n


def greedy_value(env, sol: dict) -> float:
    """Soma dos lances w_j / c_ij da solucao (objetivo do guloso global / CBAA)."""
    drones, tr, cost, _ = rodada(env)
    idx = {p["id"]: j for j, p in enumerate(tr)}
    w = np.array([priority_weight(env, p) for p in tr])
    return float(sum(w[idx[sol[d]]] / max(cost[i, idx[sol[d]]], 1.0) for i, d in enumerate(drones)
                     if sol.get(d) is not None and sol[d] in idx))


PAIR_VALUE = {"leilao|hungaro_prioridade": lap_cost, "cbaa_dist|guloso_global": greedy_value,
              "milp_wta|hungaro_wta": objective_of}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default=os.path.join(ROOT, "missions", "realista"))
    ap.add_argument("--n-scen", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=60, help="rodadas avaliadas por cenario")
    ap.add_argument("--paper", default=os.path.join(ROOT, "paper"))
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.missions, "**", "*.json"), recursive=True))
    paths = [p for p in paths if not p.endswith("manifest.json")]
    paths = paths[:: max(1, len(paths) // a.n_scen)][: a.n_scen]
    cfg = RefuelConfig(max_tracks=64, t_max=3000.0, start_frac=(0.0, 0.05), guidance="pn")   # como no artigo
    agree = {f"{x}|{y}": [] for x, y in PAIRS}
    same_val = {f"{x}|{y}": [] for x, y in PAIRS}
    gap = {k: [] for k in WTA_OBJ}
    times = {k: [] for k in SOLVERS}
    sizes = []
    for p in paths:
        sc = Scenario.from_file(p)
        env = RefuelEnv(sc, cfg, seed=10_000)
        env.skip_obs = True
        last_t, rounds = -1.0, 0
        while not env.done and rounds < a.max_rounds:
            if env.t != last_t and env.queue:
                last_t = env.t
                drones, tr, _, _ = rodada(env)
                if tr and len(drones) > 0:
                    rounds += 1
                    sols = {}
                    for k, f in SOLVERS.items():
                        t0 = time.perf_counter()
                        sols[k] = f(env)
                        times[k].append((len(drones) * len(tr), 1e3 * (time.perf_counter() - t0)))
                    sizes.append(len(drones) * len(tr))
                    for x, y in PAIRS:
                        same = [sols[x].get(d) == sols[y].get(d) for d in drones]
                        agree[f"{x}|{y}"].append(float(np.mean(same)))
                        fv = PAIR_VALUE[f"{x}|{y}"]
                        vx, vy = fv(env, sols[x]), fv(env, sols[y])
                        tol = 1e-6 * max(1.0, abs(vy)) if fv is not lap_cost else 1e-3 * len(drones)
                        same_val[f"{x}|{y}"].append(float(abs(vx - vy) <= tol))
                    objs = {k: objective_of(env, sols[k]) for k in WTA_OBJ}
                    best = max(objs.values())
                    for k in WTA_OBJ:
                        gap[k].append(objs[k] / best if best > 1e-9 else 1.0)
            env.step(METHODS["hungaro_prioridade"](env))
        print(f"{os.path.basename(p)}: {rounds} rodadas", flush=True)
    out = {"concordancia": {k: float(np.mean(v)) for k, v in agree.items()},
           "mesmo_valor": {k: float(np.mean(v)) for k, v in same_val.items()},
           "objetivo_relativo": {k: float(np.mean(v)) for k, v in gap.items()},
           "objetivo_relativo_min": {k: float(np.min(v)) for k, v in gap.items()},
           "tempo_ms": {k: {"mediana": float(np.median([t for _, t in v])),
                            "p95": float(np.percentile([t for _, t in v], 95))} for k, v in times.items()},
           "rodadas": len(sizes), "tamanho_mediano": float(np.median(sizes)), "tamanho_max": int(np.max(sizes))}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "equivalencia.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    lines = []
    for k in SOLVERS:
        conc = next((out["concordancia"][f"{x}|{y}"] for x, y in PAIRS if x == k), None)
        sv = next((out["mesmo_valor"][f"{x}|{y}"] for x, y in PAIRS if x == k), None)
        ref = next((NAMES[y] for x, y in PAIRS if x == k), "")
        rel = out["objetivo_relativo"].get(k)
        t = out["tempo_ms"][k]
        f = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
        lines.append(f"{NAMES[k]} & {ref or '--'} & {f(100 * conc, 1) if conc is not None else '--'} & "
                     f"{f(100 * sv, 1) if sv is not None else '--'} & "
                     f"{f(rel) if rel is not None else '--'} & {f(t['mediana'], 2)} & {f(t['p95'], 2)} \\\\")
    tex = r"""\begin{table*}[t]
\centering
\caption{Validação das implementações e custo por rodada de decisão: %d rodadas de %d cenários, resolvidas por todos
os algoritmos (instância mediana com %d pares interceptador--pista; máxima com %d). Mesma designação: fração dos
interceptadores designados igual à referência teoricamente equivalente; mesmo valor: fração das rodadas em que o valor
do objetivo coincide (a designação pode diferir quando há ótimos alternativos). Objetivo relativo: valor do
objetivo~\eqref{eq:F} dividido pelo melhor encontrado na rodada.}
\label{tab:equivalencia}
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}llccccc@{}}
\toprule
Algoritmo & Referência equivalente & Mesma designação (\%%) & Mesmo valor (\%%) & Objetivo relativo & Tempo mediano (ms) & Tempo p95 (ms) \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table*}
""" % (out["rodadas"], len(paths), out["tamanho_mediano"], out["tamanho_max"], "\n".join(lines))
    os.makedirs(os.path.join(a.paper, "tabelas"), exist_ok=True)
    open(os.path.join(a.paper, "tabelas", "equivalencia.tex"), "w", encoding="utf-8").write(tex)


if __name__ == "__main__":
    main()
