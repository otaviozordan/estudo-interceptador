"""Benchmark do artigo: todos os metodos (metodos/, M1--M29) sobre as missoes (padrao: missions/realista), em fases.

    python scripts/experimentos/run_benchmark.py                                   # tudo, numa pasta nova
    python scripts/experimentos/run_benchmark.py --phase heur --dir <pasta>        # M1--M22 em todos os cenarios
    python scripts/experimentos/run_benchmark.py --phase heur --methods M7 M13     # so alguns metodos
    python scripts/experimentos/run_benchmark.py --phase bt --dir <pasta>          # M23: GA no treino + avaliacao
    python scripts/experimentos/run_benchmark.py --phase learn --dir <pasta>       # M24--M29: treino + avaliacao
    python scripts/experimentos/run_benchmark.py --phase ablation --dir <pasta>    # guiagem, troca de alvo, salvas
    python scripts/experimentos/analyze_benchmark.py <pasta>                       # tabelas e figuras do artigo

Divisao treino/teste: o "split" do manifest.json (missoes realistas: Rio de Janeiro e Brasilia sao teste) ou, sem ele,
os locais em --test-places (relevo nunca visto). Os metodos sem
treino rodam em todos; os treinaveis so treinam no treino e as redes so sao avaliadas no teste. Cada episodio vira uma
linha de <pasta>/episodes.jsonl com o metodo, a variante (ablacao ou semente de treino), o cenario, a semente, as
metricas e os atributos do cenario (manifest.json). Mesmas sementes para todos os metodos (comparacao pareada).

Retomada: rodar de novo com o mesmo --dir pula o que ja esta em episodes.jsonl. Modelos treinados ficam em
modelos/ (os mesmos que ``rodar.py`` usa) e sao reaproveitados; com --limit, em <pasta>/modelos.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from metodos import METODOS, MODELOS, obter                       # noqa: E402
from metodos.base import ler_json                                  # noqa: E402
from metodos.paralelo import EpisodePool                           # noqa: E402
from metodos.treino import MISSOES_PADRAO, PARAMS, split_de, treinar   # noqa: E402
from swarm_sim.refuel.env import load_scenarios                    # noqa: E402
from swarm_sim.refuel.episodio import config_artigo                # noqa: E402

SEED0 = 10_000
ABL_METHODS = ["guloso_proximo", "hungaro_prioridade", "mmr"]
LEARNED = {"deepsets": ["M24", "M26", "M28"], "attn": ["M25", "M27", "M29"]}


class Run:
    def __init__(self, a):
        self.a = a
        self.dir = a.dir or os.path.join(ROOT, "results", "benchmark", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.modelos = a.modelos or (os.path.join(self.dir, "modelos") if a.limit else MODELOS)
        self.logf = open(os.path.join(self.dir, "log.txt"), "a", encoding="utf-8")
        man = ler_json(os.path.join(a.missions, "manifest.json"))
        self.scs = load_scenarios(a.missions, verbose=False)
        self.meta = {sc.name: man.get(sc.name, {}) for sc in self.scs}
        self.split = {sc.name: split_de(self.meta[sc.name], a.test_places) for sc in self.scs}
        if a.limit:
            self.scs = self.scs[:: max(1, len(self.scs) // a.limit)][: a.limit]
        self.train = [sc for sc in self.scs if self.split[sc.name] == "treino"]
        self.test = [sc for sc in self.scs if self.split[sc.name] == "teste"]
        self.cfg = config_artigo(a.guidance, a.amax)
        cfg_path = os.path.join(self.dir, "config.json")
        if not os.path.exists(cfg_path):
            json.dump({"args": vars(a), "refuel_config": dataclasses.asdict(self.cfg), "split": self.split},
                      open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        self.done: dict[tuple[str, str], int] = {}                  # retomada: (metodo, variante) ja avaliados
        ep_path = os.path.join(self.dir, "episodes.jsonl")
        if os.path.exists(ep_path):
            for line in open(ep_path, encoding="utf-8"):
                r = json.loads(line)
                key = (r["metodo"], r["variante"])
                self.done[key] = self.done.get(key, 0) + 1
        self.log(f"{len(self.scs)} cenarios ({len(self.train)} treino) -> {self.dir}"
                 + (f" (retomando: {len(self.done)} avaliacoes prontas)" if self.done else ""))

    def log(self, *x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        self.logf.write(s + "\n")
        self.logf.flush()

    def train_params(self) -> dict:
        return {k: getattr(self.a, k) for k in PARAMS}

    def evaluate(self, pool, method: str, spec, scs=None, variant: str = "base", over: dict | None = None,
                 runs: int | None = None):
        scs = self.scs if scs is None else scs
        t0 = time.time()
        tasks = [("eval", pool.index(sc), SEED0 + k, spec, over) for sc in scs
                 for k in range(min(max(sc.m.refuel.runs, 1), runs or 10**6))]
        if not tasks or self.done.get((method, variant), 0) >= len(tasks):
            self.log(f"  {method:24s} {variant:16s} ja avaliado: pulando")
            return
        rows = pool.map(tasks)
        self.done[(method, variant)] = len(rows)
        with open(os.path.join(self.dir, "episodes.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                r = r | {"metodo": method, "variante": variant, "split": self.split[r["scenario"]],
                         **{f"c_{k}": v for k, v in self.meta[r["scenario"]].items()}}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dano = sum(r["dano"] for r in rows) / len(rows)
        rate = sum(r["rate"] for r in rows) / len(rows)
        custo = sum(r.get("custo", r["dano"]) for r in rows) / len(rows)
        self.log(f"  {method:24s} {variant:16s} dano={dano:.3f} custo={custo:.3f} intercept.={100 * rate:5.1f}%  "
                 f"{len(rows)} ep. {time.time() - t0:.0f}s")


def phase_heur(R: Run, pool):
    R.log("== fase heur: M1--M22 (sem treino)")
    ms = [obter(k) for k in R.a.methods] if R.a.methods else list(METODOS.values())
    for m in ms:
        if m.treino is None:
            R.evaluate(pool, m.chave, ("base", m.chave))


def phase_bt(R: Run, pool):
    """M23 (Feitosa et al., 2025): coeficientes ajustados por GA nos cenarios de TREINO; avaliada em todos."""
    m = obter("M23")
    path = treinar(m, R.train, R.cfg, pool, modelos=R.modelos, log=R.log, **R.train_params())
    R.evaluate(pool, m.chave, ("bt", tuple(ler_json(path)["coef"])))


def phase_learn(R: Run, pool):
    import torch
    a = R.a
    R.log(f"== fase learn: sementes {a.train_seeds}, redes {a.kinds}")
    for s in a.train_seeds:
        for kind in a.kinds:
            for mid in LEARNED[kind]:
                m = obter(mid)
                if m.id not in ("M24", "M25") and a.ppo_updates <= 0:
                    continue
                path = treinar(m, R.train, R.cfg, pool, s, R.modelos, R.log, **R.train_params())
                sd = torch.load(path, map_location="cpu")
                R.evaluate(pool, m.chave, ("net", sd, True, kind), scs=R.test, variant=f"s{s}")


def phase_ablation(R: Run, pool):
    R.log("== fase ablation")
    n = R.a.abl_runs
    for k in ABL_METHODS:
        if n < 5:                                                   # base com os mesmos sorteios das variantes
            R.evaluate(pool, k, ("base", k), variant="base_abl", runs=n)
        for g in ("ideal", "pp", "pn", "apn"):
            if g != R.cfg.guidance:
                R.evaluate(pool, k, ("base", k), variant=f"guia_{g}", over={"guidance": g}, runs=n)
        R.evaluate(pool, k, ("base", k), variant="sem_troca", over={"retask": False}, runs=n)
    for k in ("mmr", "ga"):
        if k not in ABL_METHODS and n < 5:
            R.evaluate(pool, k, ("base", k), variant="base_abl", runs=n)
        R.evaluate(pool, k, ("base", k), variant="salva2", over={"max_per_track": 2}, runs=n)
    R.evaluate(pool, "hungaro_prioridade", ("base", "hungaro_prioridade"), variant="guia_mpc",
               over={"guidance": "mpc"}, runs=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["heur", "bt", "learn", "ablation", "all"], default="all")
    ap.add_argument("--dir", default=None, help="pasta de resultados existente (acrescenta / retoma)")
    ap.add_argument("--missions", default=MISSOES_PADRAO, help="padrao: missions/realista")
    ap.add_argument("--modelos", default=None, help="pasta dos modelos treinados (padrao: modelos/)")
    ap.add_argument("--test-places", nargs="+", default=None,
                    help="locais de teste (padrao: o split do manifest; em missions/enxame, Itajuba e Guaratingueta)")
    ap.add_argument("--methods", nargs="*", default=None, help="subconjunto dos metodos da fase heur (M7, mmr, ...)")
    ap.add_argument("--guidance", default="pn")
    ap.add_argument("--amax", type=float, default=30.0)
    ap.add_argument("--expert", default=PARAMS["expert"])
    ap.add_argument("--kinds", nargs="+", default=["deepsets", "attn"])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--il-iters", type=int, default=PARAMS["il_iters"])
    ap.add_argument("--il-episodes", type=int, default=PARAMS["il_episodes"])
    ap.add_argument("--il-keep", type=float, default=PARAMS["il_keep"], help="fracao das esperas guardadas no DAgger")
    ap.add_argument("--abl-runs", type=int, default=3, help="sorteios por missao nas ablacoes")
    ap.add_argument("--limit", type=int, default=0, help="usa so N cenarios (teste rapido)")
    ap.add_argument("--bt-pop", type=int, default=PARAMS["bt_pop"])
    ap.add_argument("--bt-gens", type=int, default=PARAMS["bt_gens"])
    ap.add_argument("--bt-scenarios", type=int, default=PARAMS["bt_scenarios"])
    ap.add_argument("--bt-runs", type=int, default=PARAMS["bt_runs"])
    ap.add_argument("--bt-seed", type=int, default=PARAMS["bt_seed"])
    ap.add_argument("--ppo-updates", type=int, default=PARAMS["ppo_updates"])
    ap.add_argument("--ppo-episodes", type=int, default=PARAMS["ppo_episodes"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    R = Run(a)
    t0 = time.time()
    with EpisodePool(R.scs, R.cfg, a.workers) as pool:
        if a.phase in ("heur", "all"):
            phase_heur(R, pool)
        if a.phase in ("bt", "all"):
            phase_bt(R, pool)
        if a.phase in ("learn", "all"):
            phase_learn(R, pool)
        if a.phase in ("ablation", "all"):
            phase_ablation(R, pool)
    R.log(f"fim ({time.time() - t0:.0f}s) -> {R.dir}")


if __name__ == "__main__":
    main()
