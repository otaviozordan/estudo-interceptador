"""Missoes realistas: enxames mistos (drones de ataque de longo alcance, iscas, municoes vagantes e FPV) contra
ativos reais de infraestrutura (OpenStreetMap), com a defesa montada SEM conhecer o ataque.

    python scripts/missoes/osm_ativos.py                 # 1o: ativos de cada cidade -> data/osm/
    python scripts/missoes/make_realista.py              # 6 cidades x 32 missoes -> missions/realista/
    python scripts/missoes/make_realista.py --places "Sao Paulo" --reps 1

Diferencas para o gerador sintetico (make_swarm.py):
* zonas prioritarias = ativos reais (usinas, aerodromos, subestacoes, agua/esgoto, telecom, industria >= 5 ha),
  escolhidos por importancia; os alvos sao esses ativos (75 % das ameacas) ou outros ativos nao defendidos;
* defesa independente do ataque: radares nos morros que melhor cobrem as AREAS DEFENDIDAS (nao as rotas) e bases de
  interceptadores junto dos ativos; nenhuma missao e descartada por cobertura;
* eixo de ameaca: as entradas ficam num setor de 60--90 graus da borda (deslocado ate 20 graus entre ondas);
* rotas planejadas sobre o relevo (caminho de menor custo): quem voa baixo segue os vales;
* classes com parametros de fontes abertas (ver CLASSES) e chegada coordenada no alvo (iscas na frente).

Delineamento: por cidade, 4 tamanhos x 4 composicoes x --reps (padrao 2) = 32 missoes; toda missao tem as quatro
classes. Treino/teste por cidade (--test-places), gravado no manifest.json ("split").
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np                                                            # noqa: E402
from scipy.ndimage import gaussian_filter                                     # noqa: E402
from scipy.sparse import coo_matrix                                           # noqa: E402
from scipy.sparse.csgraph import dijkstra                                     # noqa: E402

from make_battery import CELLS, COVER_AGL, hilltops, radar_template           # noqa: E402
from swarm_sim.mission import (AircraftSpec, DroneBase, Mission, PriorityZone, RefuelSpec, load_terrain,  # noqa: E402
                               place_origin, slug)
from swarm_sim.missionmap import save_mission_map                             # noqa: E402
from swarm_sim.radar import coverage_mask                                     # noqa: E402
from swarm_sim.routes import Waypoint                                         # noqa: E402

PLACES = ["Sao Jose dos Campos", "Taubate", "Itajuba (centro)", "Sao Paulo", "Rio de Janeiro", "Brasilia"]
TEST_PLACES = ["Rio de Janeiro", "Brasilia"]
SIZES = [12, 24, 36, 48]

# Classes de ameaca. Velocidades e alturas de fontes abertas (ver docs/missoes_realistas.md); RCS e valor sao
# hipoteses do modelo. valor = peso do dano do impacto (isca = 0); dive = inclinacao do mergulho final (dAGL/ds).
CLASSES = {
    # drone de ataque de longo alcance (tipo Shahed-136/Geran-2): 150--185 km/h, 60--4000 m, rota pre-programada
    "ataque": dict(v=(42.0, 51.0), agl_baixo=(60.0, 150.0), agl_alto=(500.0, 1500.0), p_alto=0.5, dive=1.0,
                   rcs=(0.3, 1.0), valor=1.0),
    # isca (tipo Gerbera): ~97 km/h de cruzeiro, ate ~160 km/h, assinatura parecida (as vezes ampliada)
    "isca": dict(v=(27.0, 44.0), agl_baixo=(80.0, 200.0), agl_alto=(500.0, 1200.0), p_alto=0.3, dive=0.3,
                 rcs=(0.5, 1.5), valor=0.0),
    # municao vagante (tipo Lancet-3): 80--110 km/h, orbita sobre o alvo e mergulha (~300 km/h no mergulho)
    "vagante": dict(v=(22.0, 31.0), agl_baixo=(300.0, 800.0), agl_alto=(300.0, 800.0), p_alto=0.0, dive=2.0,
                    rcs=(0.1, 0.4), valor=0.5, loiter=(400.0, 700.0)),
    # FPV de ataque: 60--80 km/h em combate (max. ~120), voo rente ao solo, pilotado (manobra e reage)
    "fpv": dict(v=(17.0, 33.0), agl_baixo=(20.0, 50.0), agl_alto=(20.0, 50.0), p_alto=0.0, dive=0.5,
                rcs=(0.02, 0.1), valor=0.25),
}
# composicao do enxame (fracoes por classe); toda missao tem pelo menos uma ameaca de cada classe.
# ~60 % ataque / 40 % iscas e a proporcao relatada em ondas reais (saturacao).
COMPOSICOES = {
    "saturacao": {"ataque": 0.45, "isca": 0.40, "vagante": 0.10, "fpv": 0.05},
    "longo_alcance": {"ataque": 0.60, "isca": 0.25, "vagante": 0.10, "fpv": 0.05},
    "tatico": {"ataque": 0.15, "isca": 0.10, "vagante": 0.40, "fpv": 0.35},
    "equilibrado": {"ataque": 0.25, "isca": 0.25, "vagante": 0.25, "fpv": 0.25},
}
FPV_EVASION = {"jink": [20, 60], "jink_dt": [2.0, 5.0], "a_max": 8.0, "speed_var": 0.1}
FPV_REACT = {"react": 300.0, "react_tgo": 3.0, "react_hold": 2.0, "a_max": 8.0}
# interceptador (tipo Sting): cruzeiro 140--170 km/h, max. ~280 km/h -> 65 m/s no engajamento
DRONE_SPEED = 65.0
# envelope de voo das ameacas conhecido pelo defensor (filtro E1): cobre todas as classes, inclusive o mergulho
ENVELOPE = dict(v_min=12.0, v_max=75.0, vh_max=60.0, agl_min=-50.0, agl_max=2500.0)
# custo de um interceptador desperdicado (sem autonomia, gasto em isca ou em voo sem uso no fim), na unidade das
# penalidades de impacto: 1/5 do impacto de uma ameaca plena num ativo nao defendido
WASTE_PENALTY = 0.2
GRID = 150.0                                                                 # grade do planejador de rotas [m]


def counts(n: int, frac: dict) -> dict:
    """Numero de ameacas por classe: pelo menos 1 de cada, o resto pelo maior resto."""
    ks = list(frac)
    base = {k: 1 for k in ks}
    rest = n - len(ks)
    raw = {k: frac[k] * rest for k in ks}
    for k in ks:
        base[k] += int(np.floor(raw[k]))
    left = n - sum(base.values())
    for k in sorted(ks, key=lambda k: -(raw[k] - np.floor(raw[k])))[:left]:
        base[k] += 1
    return base


class Planner:
    """Caminho de menor custo sobre o relevo: custo da celula = 1 + beta * altura normalizada + ruido suave. Com beta
    alto (voo baixo) a rota segue os vales; com beta baixo e quase reta. O ruido muda a cada missao."""

    def __init__(self, T, rng):
        n0, n1, e0, e1 = T.extent
        self.ns = np.arange(n0 + GRID / 2, n1, GRID)
        self.es = np.arange(e0 + GRID / 2, e1, GRID)
        N, E = np.meshgrid(self.ns, self.es, indexing="ij")
        z = T.height(N, E)
        self.h = (z - z.min()) / max(np.ptp(z), 1.0)
        nz = gaussian_filter(rng.normal(size=z.shape), 6)
        self.noise = (nz - nz.min()) / max(np.ptp(nz), 1e-9)
        n0, n1, e0, e1 = T.extent                                   # evita rotas coladas na borda do mapa
        edge = np.minimum.reduce([N - n0, n1 - N, E - e0, e1 - E])
        self.noise = self.noise + 3.0 * np.clip(1.0 - edge / 1200.0, 0.0, 1.0)
        self.shape = z.shape
        self._graphs: dict = {}
        self._dist: dict = {}

    def _graph(self, beta: float):
        if beta in self._graphs:
            return self._graphs[beta]
        c = 1.0 + beta * self.h + 0.6 * self.noise
        ny, nx = self.shape
        idx = np.arange(ny * nx).reshape(ny, nx)
        rows, cols, w = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):              # 8-vizinhanca (cada aresta uma vez)
            j0, j1 = max(0, -dj), nx - max(0, dj)
            a = idx[0:ny - di, j0:j1]
            b = idx[di:ny, j0 + dj:j1 + dj]
            L = GRID * np.hypot(di, dj)
            cw = L * 0.5 * (c.ravel()[a.ravel()] + c.ravel()[b.ravel()])
            rows += [a.ravel(), b.ravel()]
            cols += [b.ravel(), a.ravel()]
            w += [cw, cw]
        G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx,) * 2).tocsr()
        self._graphs[beta] = G
        return G

    def cell(self, p) -> int:
        i = int(np.clip(np.searchsorted(self.ns, p[0]), 0, len(self.ns) - 1))
        j = int(np.clip(np.searchsorted(self.es, p[1]), 0, len(self.es) - 1))
        return i * self.shape[1] + j

    def route(self, start, target, beta: float, spacing: float = 700.0) -> np.ndarray:
        """Pontos de controle (n, e) do inicio ao alvo, um a cada ~spacing metros."""
        key = (beta, self.cell(target))
        if key not in self._dist:
            self._dist[key] = dijkstra(self._graph(beta), indices=key[1], return_predecessors=True)[1]
        pred = self._dist[key]
        k, path = self.cell(start), []
        while k >= 0 and k != key[1] and len(path) < 10_000:
            path.append(k)
            k = pred[k]
        nx = self.shape[1]
        P = np.array([[self.ns[q // nx], self.es[q % nx]] for q in path]) if path else np.zeros((0, 2))
        step = max(1, int(round(spacing / GRID)))
        P = P[::step][1:]                                                   # sem o ponto de partida
        return P


def load_assets(place: str, T) -> list[dict]:
    path = os.path.join(ROOT, "data", "osm", slug(place.split("(")[0]) + ".json")
    if not os.path.exists(path):
        raise SystemExit(f"faltam os ativos de {place}: rode python scripts/missoes/osm_ativos.py")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    n0, n1, e0, e1 = T.extent
    out = []
    for a in d["ativos"]:
        n, e = T.from_latlon(a["lat"], a["lon"])
        if n0 + 1500 < n < n1 - 1500 and e0 + 1500 < e < e1 - 1500:
            out.append(a | {"n": float(n), "e": float(e)})
    return out


def edge_point(T, az: float, inset: float = 300.0) -> np.ndarray:
    """Ponto da borda do mapa na direcao ``az`` (graus, a partir do norte) vista do centro."""
    n0, n1, e0, e1 = T.extent
    c = np.array([(n0 + n1) / 2, (e0 + e1) / 2])
    u = np.array([np.cos(np.radians(az)), np.sin(np.radians(az))])
    half = np.array([(n1 - n0) / 2 - inset, (e1 - e0) / 2 - inset])
    t = np.min(half / np.maximum(np.abs(u), 1e-9))
    return c + t * u


def pick(rng, items, weights):
    w = np.asarray(weights, float)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def make_mission(i, rng, T, origin, place, assets, hills, cov, grid, n_ac, comp, a):
    n0, n1, e0, e1 = T.extent
    # ---------------------------------------------------------- ativos defendidos (zonas prioritarias)
    # sorteia a CATEGORIA pela importancia e depois o ativo dentro dela (por tamanho): sem isso as dezenas de
    # torres de telecom de uma cidade dominariam a escolha so por serem numerosas
    by_cat: dict[str, list[dict]] = {}
    for x in assets:
        by_cat.setdefault(x["categoria"], []).append(x)
    cats = list(by_cat)
    w_cat = [max(x["importancia"] for x in by_cat[c]) ** 3 for c in cats]
    zones: list[PriorityZone] = []
    defended: list[dict] = []
    for _ in range(int(rng.integers(2, 4))):
        for _ in range(60):
            grp = by_cat[pick(rng, cats, w_cat)]
            x = pick(rng, grp, [x["importancia"] + np.log1p(x["area_ha"]) for x in grp])
            r = float(np.clip(x["raio_m"] + 250.0, 300.0, 1000.0))
            if all(np.hypot(x["n"] - q.n, x["e"] - q.e) > r + q.radius + 300 for q in zones):
                zones.append(PriorityZone(x["n"], x["e"], r, f"P{len(zones) + 1} {x['categoria']} {x['nome'][:24]}".strip()))
                defended.append(x)
                break
    others = [x for x in assets if all(np.hypot(x["n"] - q.n, x["e"] - q.e) > q.radius + 300 for q in zones)]
    # ---------------------------------------------- defesa (independente do ataque): radares e bases
    ns, es = grid
    NN, EE = np.meshgrid(ns, es, indexing="ij")
    want = np.zeros(NN.shape)
    for q in zones:                                                  # area defendida: ate 5 km de cada ativo
        want += np.hypot(NN - q.n, EE - q.e) < 5000.0
    covered = np.zeros(NN.shape, bool)
    chosen: list[int] = []
    for _ in range(int(rng.integers(4, 8))):
        gain = np.array([(want * (c & ~covered)).sum() for c in cov])
        for c in chosen:
            gain[np.hypot(*(hills - hills[c]).T) < 1800.0] = -1
        top = np.argsort(-gain)[:3]
        top = top[gain[top] >= 0]
        if not len(top):
            break
        c = int(rng.choice(top))
        chosen.append(c)
        covered |= cov[c]
    fa = float(rng.uniform(2.0, 5.0))
    radars = [radar_template(float(hills[c][0]), float(hills[c][1]), f"R{j + 1}", fa) for j, c in enumerate(chosen)]
    bases: list[DroneBase] = []
    anchors = zones + [None] * int(rng.integers(0, 2))               # uma base por ativo (+ talvez uma central)
    for b, q in enumerate(anchors):
        for _ in range(100):
            if q is None:
                p = np.array([rng.uniform(n0 + 2500, n1 - 2500), rng.uniform(e0 + 2500, e1 - 2500)])
            else:
                ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(800.0, 2000.0)
                p = np.array([q.n + dist * np.cos(ang), q.e + dist * np.sin(ang)])
            if n0 + 800 < p[0] < n1 - 800 and e0 + 800 < p[1] < e1 - 800 and \
                    all(np.hypot(*(p - [x.n, x.e])) > 1500 for x in bases):
                bases.append(DroneBase(float(p[0]), float(p[1]), f"B{b + 1}", 0))
                break
    for d in range(int(np.ceil(1.25 * n_ac))):
        bases[d % len(bases)].drones += 1
    # ------------------------------------------------------------------ ataque
    planner = Planner(T, rng)
    az0, width = float(rng.uniform(0, 360)), float(rng.uniform(60, 90))
    n_waves = int(rng.integers(1, 4))
    wave_t = np.concatenate([[0.0], np.cumsum(rng.uniform(120.0, 300.0, n_waves - 1))])
    wave_az = az0 + rng.uniform(-20, 20, n_waves) * (np.arange(n_waves) > 0)
    cnt = counts(n_ac, COMPOSICOES[comp])
    classes = [k for k, v in cnt.items() for _ in range(v)]
    rng.shuffle(classes)
    specs, arrivals, n_hi = [], [], 0
    for k, cl in enumerate(classes):
        P = CLASSES[cl]
        w = int(rng.integers(n_waves))
        st = edge_point(T, wave_az[w] + rng.uniform(-width / 2, width / 2))
        if cl == "isca" or (rng.random() < 0.25 and others):            # alvo: ativo nao defendido
            tgt = pick(rng, others or defended, [x["importancia"] for x in (others or defended)])
            r_imp = max(tgt["raio_m"], 150.0) * 0.5
        else:                                                           # alvo: ativo defendido
            tgt = pick(rng, defended, [x["importancia"] for x in defended])
            r_imp = max(tgt["raio_m"], 150.0) * 0.3
        target = np.array([tgt["n"], tgt["e"]]) + rng.normal(0, 0.3 * r_imp, 2)
        high = rng.random() < P["p_alto"]
        n_hi += int(high)
        agl = P["agl_alto"] if high else P["agl_baixo"]
        beta = 0.3 if high else 3.0
        mids = planner.route(st, target, beta)
        if "loiter" in P:                                               # orbita sobre o alvo antes do mergulho
            R = float(rng.uniform(*P["loiter"]))
            a0 = np.arctan2(st[1] - target[1], st[0] - target[0])       # entra na orbita pelo lado de chegada
            sgn = rng.choice([-1, 1])
            orbit = [target + R * np.array([np.cos(a0 + sgn * t), np.sin(a0 + sgn * t)])
                     for t in np.linspace(0, 2 * np.pi, 9)[:-1]]
            mids = np.array([m for m in mids if np.hypot(*(m - target)) > R + 300] + orbit)
        else:
            mids = np.array([m for m in mids if np.hypot(*(m - target)) > 400]).reshape(-1, 2)
        v = P["v"]
        wps = [Waypoint(float(p[0]), float(p[1]), agl=float(np.mean(agl)), speed=float(np.mean(v)),
                        pos_radius=float(rng.uniform(100, 250)), speed_min=v[0], speed_max=v[1],
                        agl_min=agl[0], agl_max=agl[1]) for p in mids]
        wps.append(Waypoint(float(target[0]), float(target[1]), agl=0.0, speed=float(np.mean(v)),
                            pos_radius=float(r_imp * 0.3), speed_min=v[0], speed_max=v[1]))
        pts = np.vstack([st, mids.reshape(-1, 2), target])
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        # chegada coordenada: dentro da onda, chegadas em ate 60 s; iscas chegam 30--90 s antes
        arr = wave_t[w] + rng.uniform(0, 60) - (rng.uniform(30, 90) if cl == "isca" else 0.0)
        arrivals.append(arr - L / np.mean(v))
        ev = None
        if cl == "fpv":
            ev = dict(FPV_EVASION) | (FPV_REACT if rng.random() < 0.5 else {})
        specs.append(dict(name=f"{cl}{k + 1}", wps=wps, st=st, ev=ev, cl=cl, L=L,
                          rcs=float(rng.uniform(*P["rcs"])), valor=P["valor"], dive=P["dive"], v=float(np.mean(v))))
    t0 = min(arrivals)
    aircraft = [AircraftSpec(s["name"], s["wps"], {"n": float(s["st"][0]), "e": float(s["st"][1]), "heading_deg": None},
                             speed=s["v"], spline=True, rcs=s["rcs"], start_radius=200.0, evasion=s["ev"],
                             delay=float(d - t0), classe=s["cl"], valor=s["valor"], dive=s["dive"])
                for s, d in zip(specs, arrivals)]
    dur = max(a.delay + s["L"] / CLASSES[s["cl"]]["v"][0] for a, s in zip(aircraft, specs)) + 300.0
    m = Mission(name=f"{a.prefix}_{slug(place)}_{i:02d}", origin=origin, aircraft=aircraft, duration=float(dur),
                radars=radars, bases=bases, zones=zones,
                refuel=RefuelSpec(drone_speed=DRONE_SPEED, contact_radius=a.contact_radius, runs=a.runs,
                                  envelope=dict(ENVELOPE), waste_penalty=WASTE_PENALTY))
    m.validate()
    # cobertura das rotas (so informativa: nenhuma missao e descartada)
    route_pts = np.vstack([np.vstack([s["st"], *[[w.n, w.e] for w in s["wps"]]]) for s in specs])
    pi = np.clip(np.searchsorted(ns, route_pts[:, 0]), 0, len(ns) - 1)
    pj = np.clip(np.searchsorted(es, route_pts[:, 1]), 0, len(es) - 1)
    meta = {"local": place, "n_aeronaves": n_ac, "composicao": comp, "classes": cnt, "ondas": n_waves,
            "radares": len(radars), "drones": sum(b.drones for b in bases), "zonas": len(zones),
            "ativos_defendidos": [f"{x['categoria']}|{x['nome']}|{x['osm']}" for x in defended],
            "eixo_az": round(az0 % 360, 1), "setor": round(width, 1), "voo_alto": n_hi,
            "cobertura": float(covered[pi, pj].mean())}
    return m, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", nargs="+", default=PLACES)
    ap.add_argument("--test-places", nargs="+", default=TEST_PLACES)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--reps", type=int, default=2, help="repeticoes de cada tamanho x composicao por cidade")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="real")
    ap.add_argument("--contact-radius", type=float, default=3.0)
    ap.add_argument("--root", default=os.path.join(ROOT, "missions", "realista"))
    ap.add_argument("--sem-mapa", action="store_true", help="nao grava o PNG de cada missao (mais rapido)")
    a = ap.parse_args()

    man_path = os.path.join(a.root, "manifest.json")
    manifest = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    combos = [c for c in itertools.product(a.sizes, COMPOSICOES) for _ in range(a.reps)]
    for place in a.places:
        rng = np.random.default_rng([a.seed, zlib.crc32(slug(place).encode())])   # semente pelo NOME da cidade
        origin = place_origin(place)
        T = load_terrain(origin)
        assets = load_assets(place, T)
        hills = hilltops(T, k=60)
        print(f"{place}: {len(assets)} ativos no mapa, {len(hills)} morros candidatos; cobertura...", flush=True)
        cov, grid = [], None
        for n, e in hills:
            ns, es, mk = coverage_mask(T, radar_template(n, e, "c", 0.0), COVER_AGL, cells=CELLS)
            cov.append(mk)
            grid = (ns, es)
        for i, ci in enumerate(rng.permutation(len(combos))):
            n_ac, comp = combos[ci]
            m, meta = make_mission(i, rng, T, origin, place, assets, hills, cov, grid, n_ac, comp, a)
            meta["split"] = "teste" if place in a.test_places else "treino"
            path = os.path.join(a.root, slug(place), f"{m.name}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            m.save(path)
            if not a.sem_mapa:
                save_mission_map(m, T, os.path.splitext(path)[0] + ".png", COVER_AGL)
            rel = os.path.splitext(os.path.relpath(path, a.root))[0].replace(os.sep, "/")
            manifest[rel] = meta
            print(f"  {rel}: {n_ac} ({', '.join(f'{k} {v}' for k, v in meta['classes'].items())}), {comp}, "
                  f"{meta['ondas']} onda(s), {meta['radares']} radares, {meta['zonas']} ativos defendidos, "
                  f"cobertura das rotas {100 * meta['cobertura']:.0f}%", flush=True)
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{len(manifest)} missoes em {a.root}/ (manifest.json)")


if __name__ == "__main__":
    main()
