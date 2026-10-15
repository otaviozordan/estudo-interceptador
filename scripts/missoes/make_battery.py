"""Gera uma bateria de missoes de reabastecimento (presets aleatorios) para um local e salva JSON + mapa PNG.

    python scripts/missoes/make_battery.py                          # Sao Jose dos Campos, 16 missoes, 10 execucoes cada
    python scripts/missoes/make_battery.py --place "Sao Jose dos Campos" --n 24 --runs 20 --seed 1 --prefix bateria2

Cada missao varia o numero de aeronaves (1..4) e de radares (3..5) e sorteia rotas, zonas prioritarias e bases.
Os radares vao para os morros que mais cobrem as rotas (cobertura com relevo a 120 m AGL); missoes com rotas pouco
cobertas sao refeitas. Toda missao e um preset aleatorio: a cada execucao/episodio os pontos de controle sao
sorteados em discos (pos_radius), assim como velocidade (22..32 m/s), AGL (80..200 m) e o inicio (start_radius).
`refuel.runs` (--runs, editavel no construtor) diz quantas execucoes sorteadas a missao tem na avaliacao e o peso dela
no treino. Arquivos: missions/reabastecimento/<local>/<n>_radares/<n>_abastecimentos/<prefixo>_NN.json (+ .png com o mapa).
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.ndimage import maximum_filter

from swarm_sim.mission import (AircraftSpec, DroneBase, Mission, PriorityZone, RefuelSpec, load_terrain,
                               place_origin)
from swarm_sim.missionmap import save_mission_map
from swarm_sim.radar import RadarStation, coverage_mask
from swarm_sim.routes import Waypoint

COVER_AGL = 120.0
CELLS = 60


def radar_template(n: float, e: float, name: str, false_alarms: float) -> RadarStation:
    return RadarStation(n=n, e=e, name=name, height=15.0, range_m=4000.0, range_sigma=40.0,
                        false_alarms=false_alarms)


def hilltops(T, k: int = 50, sep: float = 1200.0, inset: float = 400.0) -> np.ndarray:
    """Topos de morro (maximos locais), mais altos primeiro, separados de `sep` metros."""
    w = max(3, int(800.0 / T.res) | 1)
    z = T.z
    peak = (maximum_filter(z, size=w) == z) & (z > np.percentile(z, 50))
    ii, jj = np.nonzero(peak)
    order = np.argsort(-z[ii, jj])
    out: list[tuple[float, float]] = []
    n0, n1, e0, e1 = T.extent
    for i, j in zip(ii[order], jj[order]):
        n, e = T.n0 + i * T.res, T.e0 + j * T.res
        if not (n0 + inset < n < n1 - inset and e0 + inset < e < e1 - inset):
            continue
        if all(np.hypot(n - a, e - b) >= sep for a, b in out):
            out.append((n, e))
        if len(out) >= k:
            break
    return np.array(out)


def densify(P: np.ndarray, step: float = 200.0) -> np.ndarray:
    pts = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        for s in np.arange(step, L, step):
            pts.append(a + (b - a) * s / L)
        pts.append(b)
    return np.array(pts)


def make_mission(i: int, rng, T, origin: dict, hills: np.ndarray, cov: list[np.ndarray], grid, n_ac: int,
                 n_rad: int, args) -> Mission | None:
    n0, n1, e0, e1 = T.extent
    ns, es = grid

    def inner(inset):
        return np.array([rng.uniform(n0 + inset, n1 - inset), rng.uniform(e0 + inset, e1 - inset)])

    # zonas prioritarias
    zones: list[PriorityZone] = []
    for z in range(int(rng.integers(1, 3))):
        for _ in range(50):
            c, r = inner(2500.0), float(rng.uniform(800.0, 1500.0))
            if all(np.hypot(*(c - [q.n, q.e])) > r + q.radius + 500 for q in zones):
                zones.append(PriorityZone(float(c[0]), float(c[1]), r, f"P{z + 1}"))
                break
    # rotas: borda -> 2..3 pontos internos -> impacto (60 % dentro de uma zona)
    aircraft, routes = [], []
    for k in range(n_ac):
        for _ in range(100):
            side = int(rng.integers(4))
            t = rng.uniform(0.15, 0.85)
            st = {0: (n1 - 600, e0 + t * (e1 - e0)), 1: (n0 + 600, e0 + t * (e1 - e0)),
                  2: (n0 + t * (n1 - n0), e0 + 600), 3: (n0 + t * (n1 - n0), e1 - 600)}[side]
            mids = [inner(1500.0) for _ in range(int(rng.integers(2, 4)))]
            if zones and rng.random() < 0.6:
                q = zones[int(rng.integers(len(zones)))]
                imp = np.array([q.n, q.e]) + rng.normal(0, 0.25 * q.radius, 2)
                imp_r = 0.3 * q.radius
            else:
                imp = inner(1200.0)
                if any(np.hypot(*(imp - [q.n, q.e])) < q.radius + 600 for q in zones):
                    continue
                imp_r = 300.0
            P = np.array([st, *mids, imp])
            L = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
            if 6000.0 <= L <= 16000.0:
                break
        else:
            return None
        wps = [Waypoint(float(p[0]), float(p[1]), agl=120.0, speed=25.0, pos_radius=float(rng.uniform(250, 500)),
                        speed_min=22.0, speed_max=32.0, agl_min=80.0, agl_max=200.0) for p in mids]
        wps.append(Waypoint(float(imp[0]), float(imp[1]), agl=0.0, speed=25.0, pos_radius=float(imp_r),
                            speed_min=22.0, speed_max=32.0))
        aircraft.append(AircraftSpec(f"uav{k + 1}", wps, {"n": float(st[0]), "e": float(st[1]), "heading_deg": None},
                                     speed=25.0, rcs=float(rng.uniform(0.7, 1.5)), start_radius=400.0))
        routes.append(densify(P))
    # radares: cobertura gulosa dos pontos das rotas, escolhendo entre os morros (com sorteio entre os melhores)
    pts = np.vstack(routes)
    pi = np.clip(np.searchsorted(ns, pts[:, 0]), 0, len(ns) - 1)
    pj = np.clip(np.searchsorted(es, pts[:, 1]), 0, len(es) - 1)
    hit = np.array([c[pi, pj] for c in cov])                    # (morros, pontos)
    covered = np.zeros(len(pts), bool)
    chosen: list[int] = []
    for _ in range(n_rad):
        gain = (hit & ~covered).sum(1).astype(float)
        for c in chosen:
            gain[np.hypot(*(hills - hills[c]).T) < 2000.0] = -1
        top = np.argsort(-gain)[:3]
        top = top[gain[top] >= 0]
        if not len(top):
            break
        c = int(rng.choice(top))
        chosen.append(c)
        covered |= hit[c]
    if covered.mean() < args.min_cover:
        return None
    fa = float(rng.uniform(2.0, 5.0))
    radars = [radar_template(float(hills[c][0]), float(hills[c][1]), f"R{j + 1}", fa) for j, c in enumerate(chosen)]
    # bases e drones (sobram 1..3 drones)
    bases: list[DroneBase] = []
    for b in range(int(rng.integers(2, 4))):
        for _ in range(100):
            p = inner(1000.0)
            if all(np.hypot(*(p - [q.n, q.e])) > 3000 for q in bases):
                bases.append(DroneBase(float(p[0]), float(p[1]), f"B{b + 1}", 0))
                break
    total = n_ac + int(rng.integers(1, 4))
    for d in range(total):
        bases[d % len(bases)].drones += 1
    m = Mission(name=f"{args.prefix}_{i:02d}", origin=origin, aircraft=aircraft, duration=900.0, radars=radars,
                bases=bases, zones=zones,
                refuel=RefuelSpec(contact_radius=args.contact_radius, runs=args.runs))
    m.validate()
    m.cover = float(covered.mean())                             # so para o relatorio
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default="Sao Jose dos Campos")
    ap.add_argument("--n", type=int, default=16, help="numero de missoes")
    ap.add_argument("--runs", type=int, default=10, help="execucoes (sorteios) por missao")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="bateria")
    ap.add_argument("--contact-radius", type=float, default=2.0)
    ap.add_argument("--min-cover", type=float, default=0.5, help="fracao minima das rotas coberta por radar")
    ap.add_argument("--root", default=os.path.join(ROOT, "missions", "reabastecimento"))
    a = ap.parse_args()

    origin = place_origin(a.place)
    T = load_terrain(origin)
    hills = hilltops(T)
    print(f"{a.place}: {len(hills)} morros candidatos; calculando cobertura...")
    cov, grid = [], None
    for k, (n, e) in enumerate(hills):
        ns, es, mk = coverage_mask(T, radar_template(n, e, "c", 0.0), COVER_AGL, cells=CELLS)
        cov.append(mk)
        grid = (ns, es)
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        n_ac, n_rad = 1 + i % 4, 3 + (i // 4) % 3
        for _ in range(30):
            m = make_mission(i, rng, T, origin, hills, cov, grid, n_ac, n_rad, a)
            if m is not None:
                break
        else:
            print(f"  {a.prefix}_{i:02d}: nao consegui gerar (cobertura < {a.min_cover})")
            continue
        path = m.organized_path(os.path.abspath(a.root))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m.save(path)
        save_mission_map(m, T, os.path.splitext(path)[0] + ".png", COVER_AGL)
        print(f"  {os.path.relpath(path)}: {n_ac} aeronave(s), {len(m.radars)} radares, {len(m.bases)} bases, "
              f"{sum(b.drones for b in m.bases)} drones, {len(m.zones)} zona(s), rotas {100 * m.cover:.0f}% cobertas")


if __name__ == "__main__":
    main()
