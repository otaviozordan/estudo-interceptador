"""Gera o conjunto de cenarios de enxame (dezenas de aeronaves evasivas) para o artigo de revisao/benchmark.

    python scripts/missoes/make_swarm.py                   # 6 locais x 10 missoes, 5 execucoes cada
    python scripts/missoes/make_swarm.py --places "Taubate" "Itajuba (centro)" --per-place 4 --sizes 12 24

Cada missao sorteia: N aeronaves (--sizes), um PERFIL evasivo, 1..3 ONDAS (entrada escalonada, chegadas quase
simultaneas dentro da onda), 1..3 zonas prioritarias, 4..7 radares nos morros que mais cobrem as rotas, 3..4 bases e
N + 25 % drones. Perfis (``evasion`` de cada aeronave, ver swarm_sim/refuel/evasion.py):

* ``nenhuma``     rota fixa (referencia);
* ``manobra``     zigue-zague ou curvas aleatorias + variacao de velocidade;
* ``reativa``     quebra quando um drone se aproxima;
* ``baixa``       voo baixo (30..60 m AGL) para se esconder do radar no relevo + manobra leve;
* ``mista``       mistura das anteriores por aeronave, com fintas (rota passa rente a uma zona e mergulha em outra).

Toda missao e um preset aleatorio (pontos, velocidade, AGL, inicio e manobras sorteados a cada execucao). Saida:
<root>/<local>/<n>_radares/<N>_abastecimentos/<prefixo>_<local>_NN.json + .png e <root>/manifest.json com os
atributos de cada missao (local, N, perfil, ondas) para a analise.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

from make_battery import CELLS, COVER_AGL, densify, hilltops, radar_template
from swarm_sim.mission import (AircraftSpec, DroneBase, Mission, PriorityZone, RefuelSpec, load_terrain,
                               place_origin, slug)
from swarm_sim.missionmap import save_mission_map
from swarm_sim.radar import coverage_mask
from swarm_sim.routes import Waypoint

PLACES = ["Sao Jose dos Campos", "Taubate", "Pindamonhangaba", "Campos do Jordao", "Itajuba (centro)", "Guaratingueta"]
PROFILES = ["nenhuma", "manobra", "reativa", "baixa", "mista"]

WEAVE = {"weave": [60, 150], "period": [8, 16], "a_max": 10.0, "speed_var": 0.15}
JINK = {"jink": [60, 150], "jink_dt": [2.0, 6.0], "a_max": 10.0, "speed_var": 0.15}
REACT = {"react": 400.0, "react_tgo": 4.0, "react_hold": 3.0, "a_max": 10.0}
LIGHT = {"weave": [30, 60], "period": [10, 20], "a_max": 6.0}


def evasion_for(profile: str, rng) -> tuple[dict | None, bool, bool]:
    """(evasion, voo baixo?, finta?) de uma aeronave."""
    if profile == "nenhuma":
        return None, False, False
    if profile == "manobra":
        return dict(WEAVE if rng.random() < 0.5 else JINK), False, False
    if profile == "reativa":
        return dict(REACT), False, False
    if profile == "baixa":
        return dict(LIGHT), True, False
    sub = str(rng.choice(["nenhuma", "manobra", "reativa", "baixa"]))
    ev, low, _ = evasion_for(sub, rng)
    if sub != "nenhuma" and rng.random() < 0.5:                     # manobra + reacao
        ev = (ev or {}) | REACT
    return ev, low, bool(rng.random() < 0.4)


def make_mission(i: int, rng, T, origin: dict, place: str, hills, cov, grid, n_ac: int, profile: str, a):
    n0, n1, e0, e1 = T.extent
    ns, es = grid

    def inner(inset):
        return np.array([rng.uniform(n0 + inset, n1 - inset), rng.uniform(e0 + inset, e1 - inset)])

    zones: list[PriorityZone] = []
    for z in range(int(rng.integers(1, 4))):
        for _ in range(80):
            c, r = inner(2500.0), float(rng.uniform(700.0, 1300.0))
            if all(np.hypot(*(c - [q.n, q.e])) > r + q.radius + 800 for q in zones):
                zones.append(PriorityZone(float(c[0]), float(c[1]), r, f"P{z + 1}"))
                break
    n_waves = int(rng.integers(1, 4))
    wave_t = np.concatenate([[0.0], np.cumsum(rng.uniform(90.0, 180.0, n_waves - 1))])
    aircraft, routes, n_feint = [], [], 0
    for k in range(n_ac):
        ev, low, feint = evasion_for(profile, rng)
        for _ in range(200):
            did_feint = False
            side = int(rng.integers(4))
            t = rng.uniform(0.1, 0.9)
            st = {0: (n1 - 500, e0 + t * (e1 - e0)), 1: (n0 + 500, e0 + t * (e1 - e0)),
                  2: (n0 + t * (n1 - n0), e0 + 500), 3: (n0 + t * (n1 - n0), e1 - 500)}[side]
            mids = [inner(1500.0) for _ in range(int(rng.integers(1, 3)))]
            if zones and rng.random() < 0.7:
                q = zones[int(rng.integers(len(zones)))]
                imp = np.array([q.n, q.e]) + rng.normal(0, 0.25 * q.radius, 2)
                imp_r = 0.3 * q.radius
                if feint and len(zones) > 1:                          # passa rente a outra zona antes
                    o = zones[int(rng.choice([j for j in range(len(zones)) if zones[j] is not q]))]
                    u = rng.normal(size=2)
                    mids.append(np.array([o.n, o.e]) + u / np.linalg.norm(u) * (o.radius + 400.0))
                    did_feint = True
            else:
                imp = inner(1200.0)
                if any(np.hypot(*(imp - [q.n, q.e])) < q.radius + 500 for q in zones):
                    continue
                imp_r = 300.0
            P = np.array([st, *mids, imp])
            L = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
            if 5000.0 <= L <= 14000.0:
                break
        else:
            return None
        agl = (30.0, 60.0) if low else (80.0, 200.0)
        wps = [Waypoint(float(p[0]), float(p[1]), agl=float(np.mean(agl)), speed=25.0,
                        pos_radius=float(rng.uniform(200, 400)), speed_min=22.0, speed_max=32.0,
                        agl_min=agl[0], agl_max=agl[1]) for p in mids]
        wps.append(Waypoint(float(imp[0]), float(imp[1]), agl=0.0, speed=25.0, pos_radius=float(imp_r),
                            speed_min=22.0, speed_max=32.0))
        # ondas: a aeronave entra de modo que a chegada caia perto do horario da onda (rota/25 m/s + atraso)
        w = int(rng.integers(n_waves))
        delay = float(max(0.0, wave_t[w] + rng.uniform(0.0, 40.0)))
        aircraft.append(AircraftSpec(f"uav{k + 1}", wps, {"n": float(st[0]), "e": float(st[1]), "heading_deg": None},
                                     speed=25.0, spline=True, rcs=float(rng.uniform(0.3, 1.2)), start_radius=300.0,
                                     evasion=ev, delay=delay))
        routes.append(densify(P))
        n_feint += int(did_feint)
    pts = np.vstack(routes)
    pi = np.clip(np.searchsorted(ns, pts[:, 0]), 0, len(ns) - 1)
    pj = np.clip(np.searchsorted(es, pts[:, 1]), 0, len(es) - 1)
    hit = np.array([c[pi, pj] for c in cov])
    covered = np.zeros(len(pts), bool)
    chosen: list[int] = []
    for _ in range(int(rng.integers(4, 8))):
        gain = (hit & ~covered).sum(1).astype(float)
        for c in chosen:
            gain[np.hypot(*(hills - hills[c]).T) < 1800.0] = -1
        top = np.argsort(-gain)[:3]
        top = top[gain[top] >= 0]
        if not len(top):
            break
        c = int(rng.choice(top))
        chosen.append(c)
        covered |= hit[c]
    if covered.mean() < a.min_cover:
        return None
    fa = float(rng.uniform(2.0, 5.0))
    radars = [radar_template(float(hills[c][0]), float(hills[c][1]), f"R{j + 1}", fa) for j, c in enumerate(chosen)]
    bases: list[DroneBase] = []
    for b in range(int(rng.integers(3, 5))):
        for _ in range(100):
            p = inner(1000.0)
            if all(np.hypot(*(p - [q.n, q.e])) > 2500 for q in bases):
                bases.append(DroneBase(float(p[0]), float(p[1]), f"B{b + 1}", 0))
                break
    for d in range(n_ac + int(np.ceil(0.25 * n_ac))):
        bases[d % len(bases)].drones += 1
    m = Mission(name=f"{a.prefix}_{slug(place)}_{i:02d}", origin=origin, aircraft=aircraft,
                duration=float(wave_t[-1] + 900.0), radars=radars, bases=bases, zones=zones,
                refuel=RefuelSpec(contact_radius=a.contact_radius, runs=a.runs))
    m.validate()
    meta = {"local": place, "n_aeronaves": n_ac, "perfil": profile, "ondas": n_waves, "radares": len(radars),
            "drones": sum(b.drones for b in bases), "zonas": len(zones), "cobertura": float(covered.mean()),
            "fintas": n_feint}
    return m, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", nargs="+", default=PLACES)
    ap.add_argument("--per-place", type=int, default=10, help="missoes por local")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 24, 36, 48], help="numeros de aeronaves")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="enx")
    ap.add_argument("--contact-radius", type=float, default=3.0)
    ap.add_argument("--min-cover", type=float, default=0.6)
    ap.add_argument("--root", default=os.path.join(ROOT, "missions", "enxame"))
    a = ap.parse_args()

    man_path = os.path.join(a.root, "manifest.json")
    manifest = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    rng = np.random.default_rng(a.seed)
    # delineamento balanceado: cada PAR de locais cobre todas as combinacoes tamanho x perfil (sem confundir os
    # fatores); com 6 locais x 10 missoes e 4 x 5 combinacoes, cada combinacao aparece 3 vezes, uma por par
    combos = list(itertools.product(a.sizes, PROFILES))
    for pidx, place in enumerate(a.places):
        perm = np.random.default_rng(a.seed + 1000 + pidx // 2).permutation(len(combos))
        mine = [combos[perm[((pidx % 2) * a.per_place + i) % len(combos)]] for i in range(a.per_place)]
        origin = place_origin(place)
        T = load_terrain(origin)
        hills = hilltops(T)
        print(f"{place}: {len(hills)} morros candidatos; calculando cobertura...", flush=True)
        cov, grid = [], None
        for n, e in hills:
            ns, es, mk = coverage_mask(T, radar_template(n, e, "c", 0.0), COVER_AGL, cells=CELLS)
            cov.append(mk)
            grid = (ns, es)
        for i, (n_ac, profile) in enumerate(mine):
            for _ in range(40):
                out = make_mission(i, rng, T, origin, place, hills, cov, grid, n_ac, profile, a)
                if out is not None:
                    break
            else:
                print(f"  {place} {i}: nao consegui gerar (cobertura < {a.min_cover})")
                continue
            m, meta = out
            path = m.organized_path(os.path.abspath(a.root))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            m.save(path)
            save_mission_map(m, T, os.path.splitext(path)[0] + ".png", COVER_AGL)
            rel = os.path.splitext(os.path.relpath(path, a.root))[0].replace(os.sep, "/")
            manifest[rel] = meta
            print(f"  {rel}: {n_ac} aeronaves, perfil {profile}, {meta['ondas']} onda(s), {meta['radares']} radares, "
                  f"{meta['drones']} drones, cobertura {100 * meta['cobertura']:.0f}%", flush=True)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{len(manifest)} missoes em {a.root}/ (manifest.json)")


if __name__ == "__main__":
    main()
