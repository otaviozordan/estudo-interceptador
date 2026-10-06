"""Navegacao: seguimento de rota por L1, orbitas, altitude AGL com antecipacao de terreno."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aircraft import G, wrap_pi
from .routes import Route, Waypoint
from .terrain import Terrain


@dataclass
class GuidanceCmd:
    a_lat: float          # aceleracao lateral comandada [m/s^2] (+ = direita)
    alt_cmd: float        # altitude MSL comandada [m]
    speed_cmd: float      # velocidade do ar comandada [m/s]
    xtrack: float = 0.0   # erro de rota (+ = a direita da rota) [m]
    wp_index: int = 0
    mode: str = "line"
    done: bool = False


@dataclass
class GuidanceParams:
    cruise_speed: float = 25.0
    speed_min: float = 19.0          # limites de velocidade comandavel (autoridade do profundor)
    speed_max: float = 38.0
    l1_damping: float = 0.8
    l1_period: float = 16.0          # [s]
    l1_min: float = 50.0             # [m]
    phi_max: float = np.radians(35.0)
    default_loiter_radius: float = 250.0
    # perfil vertical
    climb_gradient: float = 0.2     # inclinacao maxima planejada p/ subir [m/m]
    lookahead: float = 1800.0        # [m]
    lookahead_step: float = 30.0
    lateral_band: float = 60.0       # largura lateral (±) varrida pelo olhar-a-frente [m]
    orbit_k: float = 1.5             # ganho radial do campo vetorial de orbita
    orbit_kc: float = 0.7            # ganho de rumo na orbita [1/s]
    impact_depth: float = 40.0       # mira abaixo do solo para garantir o toque [m]
    impact_dive: float = 0.12        # inclinacao da rampa ate um waypoint no chao [m/m]
    min_agl: float = 40.0            # folga minima alem da qual o comando nunca desce [m]


class AltitudeManager:
    """Altitude MSL de referencia mantendo AGL com antecipacao do terreno.

    h_cmd = max_d [ terreno(d) + agl - g_subida * d ], d >= 0 ao longo da rota,
    i.e. a altitude que o aviao deve ter *agora* para ainda cumprir a folga em
    cada ponto a frente subindo com inclinacao g_subida. Resulta em uma subida
    suave que comeca no momento certo antes de cristas.
    """

    def __init__(self, terrain: Terrain, gp: GuidanceParams):
        self.terrain, self.gp = terrain, gp

    def command(self, ahead_ne: np.ndarray, dist: np.ndarray, agl: float) -> float:
        gp = self.gp
        # ahead_ne: (K, 2) pontos ao longo da rota
        tang = np.gradient(ahead_ne, axis=0)
        nrm = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = np.where(nrm > 1e-6, tang / np.maximum(nrm, 1e-6), np.array([1.0, 0.0]))
        right = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        h = None
        for off in (-gp.lateral_band, 0.0, gp.lateral_band):
            pts = ahead_ne + off * right
            hh = self.terrain.height(pts[:, 0], pts[:, 1])
            h = hh if h is None else np.maximum(h, hh)
        need = h + max(agl, gp.min_agl) - gp.climb_gradient * dist
        return float(np.max(need))


class Navigator:
    """Maquina de estados de rota: segmento (L1) / orbita / fim."""

    def __init__(self, route: Route, terrain: Terrain, gp: GuidanceParams | None = None):
        self.route, self.terrain = route, terrain
        self.gp = gp or GuidanceParams()
        self.alt_mgr = AltitudeManager(terrain, self.gp)
        self.idx = 0
        self.prev: np.ndarray | None = None
        self.mode = "line"
        self.orbit_center = np.zeros(2)
        self.orbit_R = 0.0
        self.orbit_dir = 1
        self.orbit_angle_acc = 0.0
        self.orbit_last_bearing = 0.0
        self.done = False
        self.completed_laps = 0

    # ----------------------------------------------------------------- utilitarios
    @property
    def wps(self) -> list[Waypoint]:
        return self.route.waypoints

    def _next_idx(self, i: int) -> int | None:
        if i + 1 < len(self.wps):
            return i + 1
        return 0 if self.route.loop else None

    @property
    def target(self) -> Waypoint:
        return self.wps[self.idx]

    def _advance(self, pos_ne, from_pos: np.ndarray | None = None):
        nxt = self._next_idx(self.idx)
        self.prev = np.array(from_pos if from_pos is not None else self.target.ne)
        if nxt is None:
            self._start_orbit(self.target.ne, self.gp.default_loiter_radius, 1, pos_ne)
            self.done = True
            self.mode = "final_orbit"
            return
        if nxt == 0:
            self.completed_laps += 1
        self.idx = nxt
        self.mode = "line"

    def _start_orbit(self, center, R, direction, pos_ne):
        self.mode = "orbit"
        self.orbit_center, self.orbit_R, self.orbit_dir = np.array(center), R, direction
        self.orbit_angle_acc = 0.0
        d = np.asarray(pos_ne) - self.orbit_center
        self.orbit_last_bearing = np.arctan2(d[1], d[0])

    # --------------------------------------------------------------- guiado lateral
    def _l1_len(self, Vg: float) -> float:
        gp = self.gp
        return max(gp.l1_min, gp.l1_damping * gp.l1_period * Vg / np.pi)

    def _line_guidance(self, pos, vel, Vg):
        if self.prev is None:
            self.prev = pos.copy()
        A, B = self.prev, self.target.ne
        AB = B - A
        L = np.linalg.norm(AB)
        if L < 1.0:
            self._advance(pos)
            return 0.0, 0.0
        t = AB / L
        rel = pos - A
        along = rel @ t
        y = rel @ np.array([-t[1], t[0]])          # + = direita da rota
        L1 = self._l1_len(Vg)
        # ponto de referencia a distancia L1 (na rota se |y|<L1, senao perpendicular)
        s_ref = along + np.sqrt(max(L1**2 - y**2, 0.0)) if abs(y) < L1 else along
        s_ref = min(s_ref, L)
        ref = A + s_ref * t
        los = ref - pos
        # angulo entre a velocidade (solo) e a linha de visada
        eta = wrap_pi(np.arctan2(los[1], los[0]) - np.arctan2(vel[1], vel[0]))
        a_lat = 2 * Vg**2 / L1 * np.sin(np.clip(eta, -np.pi / 2, np.pi / 2))

        # criterio de troca: ponto de curva (fly-by) com antecedencia pelo raio de curva
        wp = self.target
        if wp.impact:
            pass                              # ponto de impacto: nunca troca de waypoint
        elif wp.loiter_radius > 0 and wp.loiter_turns > 0:
            if np.linalg.norm(B - pos) < max(2.5 * wp.loiter_radius, 150.0):
                self._start_orbit(B, wp.loiter_radius, wp.loiter_dir, pos)
        else:
            lead = 0.0
            nxt = self._next_idx(self.idx)
            if nxt is not None:
                n2 = self.wps[nxt].ne - B
                if np.linalg.norm(n2) > 1.0:
                    dchi = abs(wrap_pi(np.arctan2(n2[1], n2[0]) - np.arctan2(t[1], t[0])))
                    Rt = Vg**2 / (G * np.tan(self.gp.phi_max * 0.8))
                    lead = min(Rt * np.tan(min(dchi, np.radians(150)) / 2), 0.45 * L)
            if along >= L - max(lead, 20.0):
                self._advance(pos)
        return a_lat, y

    def _orbit_guidance(self, pos, vel, Vg):
        d = pos - self.orbit_center
        rho_dist = np.linalg.norm(d)
        bearing = np.arctan2(d[1], d[0])
        self.orbit_angle_acc += wrap_pi(bearing - self.orbit_last_bearing) * self.orbit_dir
        self.orbit_last_bearing = bearing
        gp, R = self.gp, self.orbit_R
        # campo vetorial: rumo tangente + convergencia pelo erro radial (saturada em +-90 deg)
        chi_tan = bearing + self.orbit_dir * np.pi / 2
        chi_des = chi_tan + self.orbit_dir * np.arctan(gp.orbit_k * (rho_dist - R) / R)
        chi = np.arctan2(vel[1], vel[0])
        a_lat = (self.orbit_dir * Vg**2 / R * np.cos(np.arctan(gp.orbit_k * (rho_dist - R) / R))
                 + gp.orbit_kc * Vg * wrap_pi(chi_des - chi))
        # (rumo medido do norte para o leste cresce no sentido horario -> dir=+1 horario)
        if not self.done and self.orbit_angle_acc >= 2 * np.pi * self.target.loiter_turns:
            self._advance(pos, from_pos=pos)
        return a_lat, (rho_dist - R) * self.orbit_dir

    # ----------------------------------------------------------- olhar a frente (AGL)
    def _path_ahead(self, pos, vel):
        gp = self.gp
        dmax, step = gp.lookahead, gp.lookahead_step
        if self.mode in ("orbit", "final_orbit"):
            s = np.arange(0, min(dmax, 2 * np.pi * self.orbit_R * 1.1), step)
            d0 = pos - self.orbit_center
            th0 = np.arctan2(d0[1], d0[0])
            th = th0 + self.orbit_dir * s / self.orbit_R
            pts = self.orbit_center + self.orbit_R * np.stack([np.cos(th), np.sin(th)], axis=1)
            lead_in = abs(np.linalg.norm(d0) - self.orbit_R)     # trecho ate a circunferencia
            return np.vstack([pos, pts]), np.concatenate([[0.0], s + lead_in])
        verts = [pos.copy()]
        i, total = self.idx, 0.0
        for _ in range(len(self.wps) + 1):
            verts.append(self.wps[i].ne)
            total += np.linalg.norm(verts[-1] - verts[-2])
            nxt = self._next_idx(i)
            if total >= dmax or nxt is None:
                break
            i = nxt
        V = np.array(verts)
        seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        s = np.arange(0, min(dmax, cum[-1]) + 1e-6, step)
        if len(s) < 2:
            s = np.array([0.0, step])
        n = np.interp(s, cum, V[:, 0], right=V[-1, 0])
        e = np.interp(s, cum, V[:, 1], right=V[-1, 1])
        return np.stack([n, e], axis=1), s

    # ------------------------------------------------------------------- interface
    def update(self, pos_ne, vel_ne, Vg: float) -> GuidanceCmd:
        pos, vel = np.asarray(pos_ne, float), np.asarray(vel_ne, float)
        Vg = max(Vg, 5.0)
        if self.mode == "line":
            a_lat, y = self._line_guidance(pos, vel, Vg)
        else:
            a_lat, y = self._orbit_guidance(pos, vel, Vg)
        ahead, dist = self._path_ahead(pos, vel)
        wp = self.target
        alt = self.alt_mgr.command(ahead, dist, wp.agl if not wp.impact else self.gp.min_agl)
        if wp.impact:   # waypoint no chao: desce em rampa ate o ponto e ignora a folga
            d_wp = float(np.linalg.norm(wp.ne - pos))
            glide = float(self.terrain.height(*wp.ne)) + wp.agl - self.gp.impact_depth + self.gp.impact_dive * d_wp
            alt = min(alt, glide)
        spd = float(np.clip(wp.speed or self.gp.cruise_speed, self.gp.speed_min, self.gp.speed_max))
        amax = G * np.tan(self.gp.phi_max)
        return GuidanceCmd(float(np.clip(a_lat, -amax, amax)), alt, spd, float(y),
                           self.idx, self.mode, self.done)
