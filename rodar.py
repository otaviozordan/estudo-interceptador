"""Roda um ou mais metodos de alocacao (M1--M29) numa missao, com os modelos ja treinados de modelos/.

    python rodar.py --listar                                          # metodos e quais estao treinados
    python rodar.py --missao missions/realista/sao_paulo/real_sao_paulo_03.json --metodo M7
    python rodar.py --missao rio_de_janeiro/real_rio_de_janeiro_10 --metodo M7 M13 M17 --execucoes 5
    python rodar.py --missao <missao> --metodo todos                  # todos os disponiveis (pula nao treinados)
    python rodar.py --missao <missao> --metodo M24 --semente-treino 1 # rede da semente de treino 1
    python rodar.py --missao <missao> --metodo M7 --ver               # replay na janela
    python rodar.py --missao <missao> --metodo M7 --gif ep.gif        # grava o replay

--missao: arquivo .json, nome relativo a missions/realista ou missions/enxame (sem .json), pasta ou glob.
As execucoes usam as sementes 10000, 10001, ... (as mesmas do benchmark do artigo), entao o resultado da execucao k de
uma missao e igual ao do benchmark. Metodo: id (M7, 7) ou chave (hungaro_prioridade).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from metodos import METODOS, MODELOS, obter                         # noqa: E402
from swarm_sim.refuel.env import Scenario, load_scenarios           # noqa: E402
from swarm_sim.refuel.episodio import config_artigo, run_episode   # noqa: E402

CONJUNTOS = [os.path.join(ROOT, "missions", "realista"), os.path.join(ROOT, "missions", "enxame")]
SEED0 = 10_000


def status(m, modelos: str) -> str:
    if m.treino is None:
        return "pronto"
    if m.treino == "coef":
        return "treinado" if m.treinado(0, modelos) else "coeficientes padrao (nao ajustado)"
    seeds = sorted(int(os.path.basename(f)[1:-3]) for f in glob.glob(os.path.join(modelos, m.pasta, "s*.pt")))
    return f"treinado (sementes {', '.join(map(str, seeds))})" if seeds else "NAO treinado"


def listar(modelos: str):
    fam = None
    for m in METODOS.values():
        if m.familia != fam:
            fam = m.familia
            print(f"\n{fam}")
        print(f"  {m.id}  {m.nome:28s} {m.chave:24s} {status(m, modelos)}")
    print(f"\nmodelos em {modelos}")


def carregar_missoes(spec: str) -> list[Scenario]:
    cand = [spec] + [os.path.join(r, s) for r in CONJUNTOS for s in (spec, spec + ".json")]
    for c in cand:
        if os.path.isfile(c):
            root = next((r for r in CONJUNTOS if os.path.abspath(c).startswith(os.path.abspath(r))), None)
            return [Scenario.from_file(c, root=root)]
        if os.path.isdir(c):
            return load_scenarios(c, verbose=False)
    scs = load_scenarios(spec, verbose=False) if glob.glob(spec) else []
    if not scs:
        raise SystemExit(f"missao nao encontrada: {spec}")
    return scs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missao", help="arquivo, nome em missions/realista ou missions/enxame, pasta ou glob")
    ap.add_argument("--metodo", nargs="+", default=["M07"], help="ids/chaves dos metodos, ou 'todos'")
    ap.add_argument("--execucoes", type=int, default=1, help="execucoes sorteadas por missao (sementes 10000+k)")
    ap.add_argument("--semente", type=int, default=SEED0, help="semente da primeira execucao")
    ap.add_argument("--semente-treino", type=int, default=0, help="qual rede treinada usar (M24--M29)")
    ap.add_argument("--modelos", default=MODELOS, help="pasta dos modelos treinados")
    ap.add_argument("--guiagem", default="pn", choices=["pn", "pp", "apn", "ideal", "mpc"])
    ap.add_argument("--amax", type=float, default=30.0, help="aceleracao lateral maxima do drone [m/s^2]")
    ap.add_argument("--ver", action="store_true", help="abre o replay (1 metodo, 1 execucao)")
    ap.add_argument("--gif", help="grava o replay em .gif/.mp4 (1 metodo, 1 execucao)")
    ap.add_argument("--velocidade", type=float, default=10.0, help="segundos simulados por segundo de replay")
    ap.add_argument("--saida", help="grava os episodios em .jsonl")
    ap.add_argument("--workers", type=int, default=1, help="processos em paralelo (muitos episodios)")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()
    if a.listar or not a.missao:
        listar(a.modelos)
        if not a.missao:
            print("\nuse --missao <arquivo> --metodo M7 (veja --help)")
        return

    if a.metodo == ["todos"]:
        ms = [m for m in METODOS.values() if m.treino != "rede" or m.treinado(a.semente_treino, a.modelos)]
        skip = [m.id for m in METODOS.values() if m not in ms]
        if skip:
            print(f"(sem modelo treinado, pulados: {', '.join(skip)})")
    else:
        ms = [obter(k) for k in a.metodo]
    for m in ms:
        if not m.treinado(a.semente_treino, a.modelos):
            if m.treino == "rede":
                raise SystemExit(f"{m.id} ({m.nome}) nao treinado: falta {m.arquivo(a.semente_treino, a.modelos)}\n"
                                 f"  treine com: python scripts/treinar.py {m.id} --semente {a.semente_treino}")
            print(f"aviso: {m.id} sem coeficientes ajustados em {m.arquivo(0, a.modelos)}; usando os padrao")

    scs = carregar_missoes(a.missao)
    cfg = config_artigo(a.guiagem, a.amax)
    seeds = [a.semente + k for k in range(a.execucoes)]
    print(f"{len(scs)} missao(oes) x {len(seeds)} execucao(oes) x {len(ms)} metodo(s); guiagem {a.guiagem}")

    if a.ver or a.gif:
        if len(ms) != 1 or len(scs) != 1:
            raise SystemExit("--ver/--gif: um metodo e uma missao")
        import matplotlib
        if a.gif:
            matplotlib.use("Agg")
        from swarm_sim.refuel.env import RefuelEnv
        from swarm_sim.refuel.replay import record, save_gif, show
        m, sc = ms[0], scs[0]
        pol = m.politica(a.semente_treino, a.modelos)
        env = RefuelEnv(sc, cfg, seed=seeds[0])
        frames, ret = record(env, pol)
        s = env.summary()
        print(f"{m.id} {m.nome}: dano {s['dano']:.3f}, {s['refueled']}/{s['n_aircraft']} interceptadas, "
              f"{s['crashed']} impactos ({s['crashed_priority']} em zona prioritaria), {s['drones_usados']} drones")
        title = f"{sc.name} | {m.id} {m.nome} | semente {seeds[0]}"
        if a.gif:
            save_gif(env, frames, a.gif, title, hasattr(pol, "classify"), a.velocidade)
            print(f"salvo em {a.gif}")
        else:
            show(env, frames, title, hasattr(pol, "classify"), a.velocidade)
        return

    rows = []
    if a.workers > 1:
        from metodos.paralelo import EpisodePool
        with EpisodePool(scs, cfg, a.workers) as pool:
            for m in ms:
                t0 = time.time()
                spec = ("metodo", m.id, a.semente_treino, a.modelos)
                out = pool.map([("eval", i, sd, spec) for i in range(len(scs)) for sd in seeds])
                rows += [r | {"metodo": m.id, "chave": m.chave} for r in out]
                resumo(m, out, time.time() - t0)
    else:
        for m in ms:
            t0 = time.time()
            pol = m.politica(a.semente_treino, a.modelos)
            out = []
            for sc in scs:
                for sd in seeds:
                    r = run_episode(pol, sc, cfg, sd)
                    out.append(r)
                    if len(scs) * len(seeds) > 1:
                        print(f"    {sc.name} semente {sd}: dano {r['dano']:.3f}", flush=True)
            rows += [r | {"metodo": m.id, "chave": m.chave} for r in out]
            resumo(m, out, time.time() - t0)
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"episodios em {a.saida}")


def resumo(m, out: list[dict], dt: float):
    n = len(out)
    mean = lambda k: sum(r[k] for r in out) / n                          # noqa: E731
    used = sum(r["drones_usados"] / max(r["n_aircraft"], 1) for r in out) / n
    print(f"  {m.id} {m.nome:28s} dano={mean('dano'):.3f}  custo={mean('custo'):.3f}  "
          f"intercept.={100 * mean('rate'):5.1f}%  impactos prior.={mean('crashed_priority'):.2f}  "
          f"desperdicio={mean('desperdicio'):.1f}  drones/aeron.={used:.2f}  "
          f"ms/decisao={mean('ms_decisao'):.2f}  ({n} ep., {dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
