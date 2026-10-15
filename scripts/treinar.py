"""Treina os metodos com parametros ajustados a partir de dados e grava em modelos/ (lido por rodar.py e pelo
benchmark). So usa os cenarios de TREINO (split do manifest; padrao missions/realista), como no artigo.

    python scripts/treinar.py M23                       # arvore com utilidade: GA dos coeficientes
    python scripts/treinar.py M24 M25 --semente 0 1 2   # imitacao (DeepSets e atencao), tres sementes
    python scripts/treinar.py M26                       # imitacao + PPO (treina o M24 antes, se faltar)
    python scripts/treinar.py todos --semente 0         # M23--M29
    python scripts/treinar.py M28 --refazer             # treina de novo mesmo se ja existir

Parametros padrao = os do artigo (metodos/treino.py: PARAMS); para um teste rapido, reduza com --il-iters,
--il-episodes, --ppo-updates, --ppo-episodes, --bt-pop, --bt-gens e salve noutra pasta com --modelos.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from metodos import METODOS, MODELOS, obter                         # noqa: E402
from metodos.base import ler_json                                   # noqa: E402
from metodos.paralelo import EpisodePool                            # noqa: E402
from metodos.treino import MISSOES_PADRAO, PARAMS, split_de, treinar   # noqa: E402
from swarm_sim.refuel.env import load_scenarios                     # noqa: E402
from swarm_sim.refuel.episodio import config_artigo                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metodos", nargs="+", help="M23..M29 ou 'todos'")
    ap.add_argument("--semente", type=int, nargs="+", default=[0], help="sementes de treino (redes)")
    ap.add_argument("--missions", default=MISSOES_PADRAO, help="padrao: missions/realista")
    ap.add_argument("--test-places", nargs="+", default=None, help="padrao: o split do manifest")
    ap.add_argument("--modelos", default=MODELOS)
    ap.add_argument("--refazer", action="store_true", help="treina de novo mesmo se o modelo ja existe")
    ap.add_argument("--guidance", default="pn")
    ap.add_argument("--amax", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for k, v in PARAMS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()

    ms = [m for m in METODOS.values() if m.treino] if a.metodos == ["todos"] else [obter(k) for k in a.metodos]
    bad = [m.id for m in ms if not m.treino]
    if bad:
        raise SystemExit(f"{', '.join(bad)} nao tem treino (rode direto com rodar.py)")
    man = ler_json(os.path.join(a.missions, "manifest.json"))
    scs = load_scenarios(a.missions, verbose=False)
    train = [sc for sc in scs if split_de(man.get(sc.name, {}), a.test_places) == "treino"]
    cfg = config_artigo(a.guidance, a.amax)
    print(f"{len(train)} cenarios de treino (de {len(scs)}); modelos em {a.modelos}")
    params = {k: getattr(a, k) for k in PARAMS}
    t0 = time.time()
    with EpisodePool(train, cfg, a.workers) as pool:
        for m in ms:
            for s in (a.semente if m.treino == "rede" else [0]):
                path = treinar(m, train, cfg, pool, s, a.modelos, refazer=a.refazer, **params)
                print(f"{m.id} {m.nome}: {path}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
