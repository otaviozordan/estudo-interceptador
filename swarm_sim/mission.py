"""Arquivo de missao (JSON): origem do DEM, aeronaves, posicao inicial e waypoints.

Formato (coordenadas locais NED em metros, origem em origin.lat/lon):
{
  "version": 1,
  "name": "demo",
  "origin": {"lat": -22.9, "lon": -45.4, "size_km": 30},
  "duration": 900, "dt": 0.02,
  "wind": {"steady_ne": [0, 0], "gust_sigma": 0.0},
  "aircraft": [
    {"name": "uav1", "speed": 25, "loop": false, "start": {"n": 0, "e": 0, "heading_deg": 90},
     "waypoints": [{"n": 100, "e": 200, "agl": 120, "speed": null,
                    "loiter_radius": 0, "loiter_turns": 0, "loiter_dir": 1}]}
  ]
}
Waypoint com agl <= 0 e um ponto no chao: a aeronave mergulha e colide com o terreno.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .gusts import GustZone, gust_from_dict
from .radar import RadarStation, radar_from_dict
from .routes import Route, Waypoint, expand_spline
from .simulation import Simulation, make_agent
from .terrain import Terrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ORIGIN = {"lat": -22.9, "lon": -45.4, "size_km": 30.0}

# Locais prontos (lat, lon, lado sugerido [km]); o DEM local cobre ~ -24..-22 S, -46..-45 W
PLACES = {
    "Itajuba (centro)": (-22.4226, -45.4529, 12.0),
    "Itajuba (aeroporto SBIT)": (-22.4089, -45.4261, 10.0),
    "Itajuba e regiao": (-22.4226, -45.4529, 40.0),
    "Pouso Alegre": (-22.2550, -45.9280, 16.0),            # centro + aeroporto SNZA
    "Santa Rita do Sapucai": (-22.2519, -45.7028, 10.0),   # centro + Inatel
    "Campos do Jordao": (-22.7390, -45.5910, 12.0),
    "Taubate": (-23.0300, -45.5600, 12.0),
    "Pindamonhangaba": (-22.9230, -45.4620, 12.0),
    "Guaratingueta": (-22.8160, -45.1930, 12.0),
    "Sao Jose dos Campos": (-23.1800, -45.8900, 14.0),
    "Sao Jose dos Campos (ITA / INPE / aeroporto)": (-23.2150, -45.8680, 10.0),
    "Vale do Paraiba (padrao)": (-22.9000, -45.4000, 30.0),
}


def place_origin(name: str) -> dict:
    """Origem do mapa para um local de PLACES (busca sem acento/caixa, aceita trecho do nome)."""
    key = name.lower().replace("á", "a").replace("é", "e").replace("ã", "a").replace("ó", "o").replace("í", "i")
    for k, (lat, lon, size) in PLACES.items():
        if key in k.lower():
            return {"lat": lat, "lon": lon, "size_km": size}
    raise KeyError(f"local desconhecido: {name!r}. Opcoes: {', '.join(PLACES)}")


def slug(text: str) -> str:
    """Nome de pasta/arquivo: minusculas, sem acento, espacos -> '_'."""
    import unicodedata
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in t)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "missao"


def place_slug(origin: dict, tol_deg: float = 0.03) -> str:
    """Local da missao para a pasta: a cidade do local pronto mais proximo (sem detalhe entre parenteses nem
    'e regiao', ex.: 'itajuba') ou lat/lon."""
    best, dmin = None, tol_deg
    for k, (lat, lon, _size) in PLACES.items():
        d = max(abs(lat - origin["lat"]), abs(lon - origin["lon"]))
        if d <= dmin:
            best, dmin = k, d
    if best is None:
        return f"lat{origin['lat']:.2f}_lon{origin['lon']:.2f}".replace("-", "m")
    return slug(best.split("(")[0].replace(" e regiao", ""))


def _plural(n: int, one: str, many: str) -> str:
    return f"{n}_{one if n == 1 else many}"


def _disc(rng, radius: float) -> np.ndarray:
    """Ponto uniforme num disco de raio `radius` (deslocamento n, e)."""
    r, th = radius * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(th), r * np.sin(th)])


_WP_FIELDS = {f.name for f in fields(Waypoint)}


def _waypoint(d: dict) -> Waypoint:
    """Waypoint a partir do JSON, ignorando campos desconhecidos (arquivos de outras versoes)."""
    return Waypoint(**{k: v for k, v in d.items() if k in _WP_FIELDS})


@dataclass
class DroneBase:
    """Ponto de lancamento de drones de reabastecimento (cada drone reabastece uma unica aeronave)."""
    n: float
    e: float
    name: str = "B1"
    drones: int = 3


@dataclass
class PriorityZone:
    """Area prioritaria (circular): a queda de uma aeronave nao reabastecida dentro dela e penalizada com
    `crash_penalty_priority`; fora de qualquer zona, com a penalidade branda `crash_penalty_other`."""
    n: float
    e: float
    radius: float = 1500.0
    name: str = "P1"


@dataclass
class RefuelSpec:
    """Parametros dos drones e do problema de reabastecimento (comuns a toda a missao)."""
    drone_speed: float = 42.0          # [m/s]
    endurance: float = 420.0           # autonomia do drone [s]
    contact_radius: float = 60.0       # distancia de contato para reabastecer [m]
    sensor_range: float = 300.0        # alcance do sensor proprio do drone (fase final do rendezvous) [m]
    crash_penalty_priority: float = 10.0   # queda dentro de uma zona prioritaria (forte)
    crash_penalty_other: float = 1.0       # queda fora das zonas prioritarias (branda)
    runs: int = 10                     # execucoes (sorteios) desta missao: episodios de avaliacao e peso no treino


@dataclass
class AircraftSpec:
    name: str
    waypoints: list[Waypoint] = field(default_factory=list)
    start: dict | None = None          # {"n","e","heading_deg"[,"agl"]}; None = automatico
    speed: float = 25.0
    loop: bool = False
    spline: bool = False               # suaviza a rota por spline (Catmull-Rom) ao simular
    rcs: float = 1.0                   # secao reta radar media [m2]
    start_radius: float = 0.0          # posicao inicial sorteada num disco deste raio (preset de treino) [m]

    def route_waypoints(self, spacing: float = 150.0) -> list[Waypoint]:
        """Waypoints efetivos: os de controle, ou a spline densificada."""
        if not self.spline or len(self.waypoints) < 2 and not self.start:
            return self.waypoints
        s = (self.start["n"], self.start["e"]) if self.start else None
        return expand_spline(self.waypoints, s, self.loop, spacing)


@dataclass
class Mission:
    name: str = "missao"
    origin: dict = field(default_factory=lambda: dict(DEFAULT_ORIGIN))
    aircraft: list[AircraftSpec] = field(default_factory=list)
    duration: float = 900.0
    dt: float = 0.02
    wind_steady_ne: tuple[float, float] = (0.0, 0.0)
    gust_sigma: float = 0.0
    gusts: list[GustZone] = field(default_factory=list)   # rajadas localizadas no mapa
    radars: list[RadarStation] = field(default_factory=list)   # estacoes de radar (unica fonte de posicao)
    bases: list[DroneBase] = field(default_factory=list)       # bases de lancamento dos drones de reabastecimento
    zones: list[PriorityZone] = field(default_factory=list)    # areas prioritarias (penalidade de queda forte)
    refuel: RefuelSpec = field(default_factory=RefuelSpec)

    # ---------------------------------------------------------------- E/S
    def to_dict(self) -> dict:
        return {
            "version": 1, "name": self.name, "origin": self.origin,
            "duration": self.duration, "dt": self.dt,
            "wind": {"steady_ne": list(self.wind_steady_ne), "gust_sigma": self.gust_sigma},
            "gusts": [asdict(g) for g in self.gusts],
            "radars": [asdict(r) for r in self.radars],
            "bases": [asdict(b) for b in self.bases],
            "zones": [asdict(z) for z in self.zones],
            "refuel": asdict(self.refuel),
            "aircraft": [{"name": a.name, "speed": a.speed, "loop": a.loop, "spline": a.spline, "rcs": a.rcs,
                          "start": a.start, "start_radius": a.start_radius,
                          "waypoints": [asdict(w) for w in a.waypoints]} for a in self.aircraft],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Mission":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        wind = d.get("wind", {})
        m = cls(name=d.get("name", os.path.splitext(os.path.basename(path))[0]),
                origin={**DEFAULT_ORIGIN, **d.get("origin", {})},
                duration=float(d.get("duration", 900.0)), dt=float(d.get("dt", 0.02)),
                wind_steady_ne=tuple(wind.get("steady_ne", (0.0, 0.0))),
                gust_sigma=float(wind.get("gust_sigma", 0.0)),
                gusts=[gust_from_dict(g) for g in d.get("gusts", [])],
                radars=[radar_from_dict(r) for r in d.get("radars", [])],
                bases=[DroneBase(**{k: v for k, v in b.items() if k in {f.name for f in fields(DroneBase)}})
                       for b in d.get("bases", [])],
                zones=[PriorityZone(**{k: v for k, v in z.items() if k in {f.name for f in fields(PriorityZone)}})
                       for z in d.get("zones", [])],
                refuel=RefuelSpec(**{k: v for k, v in d.get("refuel", {}).items()
                                     if k in {f.name for f in fields(RefuelSpec)}}))
        for a in d.get("aircraft", []):
            m.aircraft.append(AircraftSpec(
                a["name"], [_waypoint(w) for w in a.get("waypoints", [])], a.get("start"),
                float(a.get("speed", 25.0)), bool(a.get("loop", False)),
                bool(a.get("spline", False)), float(a.get("rcs", 1.0)), float(a.get("start_radius", 0.0))))
        return m

    # ------------------------------------------------- presets aleatorios
    @property
    def is_preset(self) -> bool:
        """Missao com parametros sorteados a cada execucao (raio de ponto, faixa de velocidade/AGL)."""
        return any(a.start_radius > 0 or any(w.randomized for w in a.waypoints) for a in self.aircraft)

    def randomized(self, rng: np.random.Generator | int | None = None) -> "Mission":
        """Instancia concreta do preset: sorteia posicao (disco), velocidade e AGL de cada ponto de controle e a
        posicao inicial. Rumo inicial automatico (aponta para o 1o waypoint sorteado). Nao altera a missao original."""
        import copy
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        m = copy.deepcopy(self)
        for a in m.aircraft:
            for w in a.waypoints:
                if w.pos_radius > 0:
                    w.n, w.e = (np.array([w.n, w.e]) + _disc(rng, w.pos_radius)).tolist()
                if w.speed_min is not None and w.speed_max is not None:
                    w.speed = float(rng.uniform(min(w.speed_min, w.speed_max), max(w.speed_min, w.speed_max)))
                if not w.impact and w.agl_min is not None and w.agl_max is not None:
                    w.agl = float(rng.uniform(min(w.agl_min, w.agl_max), max(w.agl_min, w.agl_max)))
            if a.start and a.start_radius > 0:
                a.start = dict(a.start)
                a.start["n"], a.start["e"] = (np.array([a.start["n"], a.start["e"]]) + _disc(rng, a.start_radius)).tolist()
                a.start["heading_deg"] = None
            if a.waypoints:
                a.speed = a.waypoints[0].speed or a.speed
        return m

    # ------------------------------------------------- organizacao em pastas
    def organized_path(self, root: str | None = None) -> str:
        """missions/<local>/<n>_radar(es)/<n>_abastecimento(s)/<nome>.json (abastecimentos = aeronaves)."""
        root = root or os.path.join(ROOT, "missions")
        radars = sum(1 for r in self.radars if r.enabled)
        return os.path.join(root, place_slug(self.origin), _plural(radars, "radar", "radares"),
                            _plural(len(self.aircraft), "abastecimento", "abastecimentos"), f"{slug(self.name)}.json")

    # ---------------------------------------------------------- validacao
    def validate(self):
        names = [a.name for a in self.aircraft]
        if len(set(names)) != len(names):
            raise ValueError("nomes de aeronaves duplicados")
        rn = [r.name for r in self.radars]
        if len(set(rn)) != len(rn):
            raise ValueError("nomes de radares duplicados")
        zn = [z.name for z in self.zones]
        if len(set(zn)) != len(zn):
            raise ValueError("nomes de zonas prioritarias duplicados")
        bn = [b.name for b in self.bases]
        if len(set(bn)) != len(bn):
            raise ValueError("nomes de bases duplicados")
        for a in self.aircraft:
            if not a.waypoints:
                raise ValueError(f"aeronave '{a.name}' sem waypoints")

    # -------------------------------------------------------------- simulacao
    def build(self, terrain: Terrain) -> Simulation:
        self.validate()
        agents = []
        for a in self.aircraft:
            wps = a.route_waypoints()
            first = wps[0]
            if a.start:
                n0, e0 = a.start["n"], a.start["e"]
                hdg = a.start.get("heading_deg")
            else:
                n0, e0, hdg = first.n - 1500.0, first.e, None
            if hdg is None:
                hdg = float(np.degrees(np.arctan2(first.e - e0, first.n - n0))) if (
                    abs(first.n - n0) + abs(first.e - e0) > 1) else 0.0
            agl0 = (a.start or {}).get("agl", first.agl if first.agl > 0 else 100.0)
            route = Route(a.name, wps, loop=a.loop,
                          kind="impacto" if any(w.impact for w in wps) else "missao")
            agents.append(make_agent(a.name, route, terrain, (n0, e0), hdg, start_agl=agl0,
                                     cruise_speed=float(np.clip(first.speed or a.speed, 19.0, 38.0)), dt=self.dt))
            agents[-1].rcs = a.rcs
        return Simulation(terrain, agents, dt=self.dt,
                          wind_steady=(self.wind_steady_ne[0], self.wind_steady_ne[1], 0.0),
                          gust_sigma=self.gust_sigma, gusts=self.gusts, radars=self.radars)


def load_terrain(origin: dict | None = None, synthetic: bool = False, res: float = 30.0) -> Terrain:
    """DEM Copernicus em data/dem/*.tif (ou fractal sintetico se nao houver/pedido)."""
    o = {**DEFAULT_ORIGIN, **(origin or {})}
    tifs = sorted(glob.glob(os.path.join(ROOT, "data", "dem", "*.tif")))
    if synthetic or not tifs:
        return Terrain.synthetic(size_km=o["size_km"], res=res)
    return Terrain.from_geotiffs(tifs, o["lat"], o["lon"], size_km=o["size_km"], res=res,
                                 cache_dir=os.path.join(ROOT, "data", "dem"))
