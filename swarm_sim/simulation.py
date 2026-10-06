"""Laco de simulacao multi-aeronave sobre um DEM, com deteccao de colisao."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aircraft import AircraftParams, FixedWing, Wind
from .autopilot import Autopilot, AutopilotGains
from .guidance import GuidanceParams, Navigator
from .gusts import gust_wind
from .radar import RadarNetwork
from .routes import Route
from .terrain import Terrain

LOG_FIELDS = ["t", "n", "e", "alt", "terrain", "agl", "phi", "theta", "psi", "Va", "Vg", "hdot",
              "alpha", "de", "da", "dr", "dt", "alt_cmd", "xtrack", "wp",
              # comandos do piloto automatico (os de/da/dr/dt acima sao as posicoes reais dos atuadores)
              "de_c", "da_c", "dr_c", "dt_c", "phi_cmd", "speed_cmd", "hdot_cmd", "a_lat", "mode",
              "p", "q", "r"]
MODE_CODES = {"line": 0, "orbit": 1, "final_orbit": 2}


@dataclass
class Agent:
    name: str
    aircraft: FixedWing
    navigator: Navigator
    autopilot: Autopilot
    cruise_speed: float
    alive: bool = True
    crash_time: float | None = None
    crash_pos: tuple[float, float, float] | None = None
    crash_cause: str = ""
    wind: Wind = field(default_factory=Wind)
    rcs: float = 1.0          # secao reta radar [m2] (alcance dos radares e para 1 m2)


def make_agent(name: str, route: Route, terrain: Terrain, start_ne, heading_deg: float,
               start_agl: float | None = None, cruise_speed: float = 25.0, dt: float = 0.02,
               params: AircraftParams | None = None, gp: GuidanceParams | None = None,
               gains: AutopilotGains | None = None) -> Agent:
    ac = FixedWing(params)
    start_agl = route.waypoints[0].agl if start_agl is None else start_agl
    alt0 = float(terrain.height(*start_ne)) + start_agl
    trim = ac.reset(start_ne[0], start_ne[1], alt0, np.radians(heading_deg), cruise_speed)
    gp = gp or GuidanceParams(cruise_speed=cruise_speed)
    nav = Navigator(route, terrain, gp)
    ap = Autopilot(ac, trim, dt, gains)
    return Agent(name, ac, nav, ap, cruise_speed)


class Simulation:
    def __init__(self, terrain: Terrain, agents: list[Agent], dt: float = 0.02,
                 wind_steady=(0.0, 0.0, 0.0), gust_sigma: float = 0.0, seed: int = 0,
                 min_separation: float = 30.0, gusts=(), radars=(),
                 stop_on_all_crashed: bool = True):
        self.terrain, self.agents, self.dt = terrain, agents, dt
        for k, a in enumerate(agents):                 # rajada independente por aeronave
            a.wind = Wind(wind_steady, gust_sigma, seed=seed + k)
        self.min_sep = min_separation
        self.gusts = list(gusts)                       # zonas de rajada (swarm_sim.gusts.GustZone)
        self.radar_net = RadarNetwork(list(radars), terrain, seed) if radars else None
        self.t = 0.0
        self.events: list[dict] = []
        self._buf = {a.name: [] for a in agents}
        self.stop_on_all_crashed = stop_on_all_crashed
        self._pair_conflict: set[tuple[str, str]] = set()
        self.min_pair_dist = np.inf

    # ------------------------------------------------------------------ passo
    def _log(self, a: Agent, cmd):
        ac = a.aircraft
        phi, theta, psi = ac.euler
        Va, alpha, _ = ac.air_data()
        vn = ac.vel_ned
        n, e = ac.x[0], ac.x[1]
        th = float(self.terrain.height(n, e))
        self._buf[a.name].append([
            self.t, n, e, ac.alt, th, ac.alt - th, phi, theta, psi, Va, np.hypot(vn[0], vn[1]),
            -vn[2], alpha, *ac.act, cmd.alt_cmd, cmd.xtrack, cmd.wp_index,
            ac.u.de, ac.u.da, ac.u.dr, ac.u.dt, a.autopilot.phi_cmd, a.autopilot.v_ref,
            a.autopilot.hdot_cmd, cmd.a_lat, MODE_CODES.get(cmd.mode, 0), *ac.x[10:13]])

    def step(self):
        dt = self.dt
        for a in self.agents:
            if not a.alive:
                continue
            ac = a.aircraft
            vn = ac.vel_ned
            cmd = a.navigator.update(ac.x[0:2], vn[0:2], float(np.hypot(vn[0], vn[1])))
            ac.u = a.autopilot.update(cmd)
            Va = ac.air_data()[0]
            p_before = np.array([ac.x[0], ac.x[1], ac.alt])
            wind = a.wind.step(Va, dt)
            if self.gusts:
                wind = wind + gust_wind(self.gusts, ac.x[0], ac.x[1])
            ac.step(dt, wind)
            p_after = np.array([ac.x[0], ac.x[1], ac.alt])
            self._log(a, cmd)
            # colisao com o terreno (inclui interseccao do segmento percorrido)
            if (self.terrain.agl(p_after[0], p_after[1], p_after[2]) <= 0.0
                    or self.terrain.segment_collision(p_before, p_after)):
                self._crash(a, "terreno")
            elif not self.terrain.inside(p_after[0], p_after[1]):
                self._crash(a, "fora_do_mapa")
        self._check_pairs()
        if self.radar_net:
            self.radar_net.step(self.t, self.agents)
        self.t += dt

    def _crash(self, a: Agent, cause: str):
        a.alive = False
        a.crash_time, a.crash_cause = self.t, cause
        a.crash_pos = tuple(a.aircraft.x[0:2]) + (a.aircraft.alt,)
        self.events.append({"t": self.t, "agent": a.name, "event": f"colisao:{cause}"})

    def _check_pairs(self):
        live = [a for a in self.agents if a.alive]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                pa, pb = live[i].aircraft.pos, live[j].aircraft.pos
                d = float(np.linalg.norm(pa - pb))
                self.min_pair_dist = min(self.min_pair_dist, d)
                key = (live[i].name, live[j].name)
                if d < self.min_sep and key not in self._pair_conflict:
                    self._pair_conflict.add(key)
                    self.events.append({"t": self.t, "agent": f"{key[0]}/{key[1]}",
                                        "event": f"separacao<{self.min_sep:g}m ({d:.1f}m)"})

    def run(self, duration: float, progress: bool = False):
        n = int(round(duration / self.dt))
        for k in range(n):
            self.step()
            if progress and k % int(60 / self.dt) == 0:
                print(f"  t={self.t:6.0f}s  " + "  ".join(
                    f"{a.name}:{'ok' if a.alive else 'X'}" for a in self.agents), flush=True)
            if self.stop_on_all_crashed and not any(a.alive for a in self.agents):
                break
        return self.results()

    # -------------------------------------------------------------- resultados
    def results(self) -> dict[str, dict[str, np.ndarray]]:
        out = {}
        for name, rows in self._buf.items():
            arr = np.array(rows) if rows else np.zeros((0, len(LOG_FIELDS)))
            out[name] = {f: arr[:, i] for i, f in enumerate(LOG_FIELDS)}
        return out

    def summary(self) -> list[dict]:
        res, rows = self.results(), []
        for a in self.agents:
            r = res[a.name]
            if len(r["t"]) == 0:
                continue
            agl_err = r["agl"] - np.array([a.navigator.wps[int(w)].agl for w in r["wp"]])
            rows.append({
                "agent": a.name, "rota": a.navigator.route.kind, "sobreviveu": a.alive,
                "causa": a.crash_cause, "t_fim": float(r["t"][-1]),
                "dist_km": float(np.sum(np.hypot(np.diff(r["n"]), np.diff(r["e"]))) / 1000),
                "agl_min": float(r["agl"].min()), "agl_medio": float(r["agl"].mean()),
                "agl_rmse": float(np.sqrt(np.mean(agl_err**2))),
                "xtrack_rms": float(np.sqrt(np.mean(r["xtrack"]**2))),
                "Va_min": float(r["Va"].min()), "Va_max": float(r["Va"].max()),
                "phi_max_deg": float(np.degrees(np.abs(r["phi"]).max())),
                "alpha_max_deg": float(np.degrees(np.abs(r["alpha"]).max())),
                "voltas": a.navigator.completed_laps, "rota_concluida": a.navigator.done,
            })
        return rows
