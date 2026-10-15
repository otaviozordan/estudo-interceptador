"""Ambiente de reabastecimento construido a partir de uma missao do construtor.

* As aeronaves seguem as rotas da missao (cinematica de ponto, sem 6-DOF) e tem combustivel infinito. O ULTIMO
  PONTO da rota e um impacto no solo: se chegarem a ele sem ter sido reabastecidas, caem. Rotas em loop nunca caem.
* Os drones saem das bases da missao; cada drone reabastece UMA aeronave e cada aeronave recebe UM drone
  (ao ser reabastecida ela sai da lista da simulacao).
* Penalidade de queda: forte (`crash_penalty_priority`) se o ponto de queda esta numa zona prioritaria da missao,
  branda (`crash_penalty_other`) fora delas.
* Os drones so conhecem as pistas do ``RadarNetwork`` (``picture``): posicao/velocidade ruidosas, sem identidade
  verdadeira e so onde ha cobertura de radar, alem das zonas prioritarias (informacao de comando). A posicao
  verdadeira so decide o contato, a queda e as metricas.
* Rendezvous: o drone voa para o ponto de interceptacao da pista de radar ate alguma aeronave entrar no alcance
  do seu sensor proprio (``refuel.sensor_range``); dai segue a posicao verdadeira dela e, ao chegar a
  ``refuel.contact_radius``, reabastece na hora.

O decisor ve um MDP sequencial: a cada passo ha um "drone atual" e a acao e escolher uma pista (slot 0..K-1) ou
o slot K. Decidem os drones ociosos na base (K = esperar) e os drones em transito (K = manter a pista atual), estes a
cada ``retask_dt`` e logo que a pista perseguida some do radar: podem abandonar uma pista perdida/falsa e assumir
outra mais proxima, no lugar de lancar um drone novo. Na rodada, os em voo decidem antes dos da base. O tempo so
avanca quando todos os drones da rodada decidiram.
Legitimidade: a maioria das pistas pode ser falso alarme. Cada pista tem caracteristicas so de radar que independem
do drone (``legit_feats``: velocidade, razao de subida, AGL, incerteza, idade, vida, medidas, distancia ao radar)
para decidir se o sinal e uma aeronave; ``plausible`` e a regra fisica (envelope de voo) usada pelas heuristicas.
O rotulo verdadeiro (``obs["legit"]``) so serve para treinar o classificador e para as metricas.
Recompensa: +1 por reabastecimento, -penalidade por queda e um termo pequeno de tempo
-(aeronaves na lista) * dt / (N * 300).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from ..mission import Mission, load_terrain
from ..radar import RadarNetwork
from ..routes import Waypoint
from .evasion import closest_approach, guidance_step, make_evasion, turn_toward

TRACK_FEATS = 17
GLOB_FEATS = 7
LEGIT_FEATS = 10
PLAUSIBLE = dict(v_min=15.0, v_max=42.0, vh_max=8.0, agl_min=-20.0, agl_max=500.0)   # envelope de voo (regra)
REFUEL_REWARD = 1.0
IMPACT_DIVE = 0.12      # inclinacao da rampa final ate o impacto (igual a GuidanceParams.impact_dive)


@dataclass
class RefuelConfig:
    max_tracks: int = 24               # slots de pistas na observacao
    dt: float = 1.0                    # passo do mundo [s]
    decision_dt: float = 2.0           # tempo entre rodadas de decisao [s]
    t_max: float = 900.0               # limite do episodio [s] (limitado pela duracao da missao)
    start_frac: tuple[float, float] = (0.0, 0.5)   # sorteio do ponto de partida na rota (fracao do comprimento)
    speed_jitter: float = 0.08
    cover_radius: float = 500.0        # pista a menos disso de uma ja designada conta como "coberta"
    retask_dt: float = 10.0            # periodo de reavaliacao do alvo de um drone em transito [s]
    guidance: str = "ideal"            # "ideal" (rumo instantaneo, modelo original) | "pp" | "pn" | "apn"
    drone_amax: float = 30.0           # aceleracao lateral maxima do drone (guiagem nao ideal) [m/s2]
    pn_gain: float = 4.0               # constante de navegacao N (pn/apn)
    terminal_dt: float = 0.1           # subpasso perto das aeronaves (guiagem nao ideal) [s]
    max_per_track: int = 1             # drones simultaneos por pista (>1 permite salva)
    retask: bool = True                # drones em voo podem trocar de pista (False: so decidem no lancamento)


def intercept_time(rel: np.ndarray, vel: np.ndarray, speed: float) -> np.ndarray:
    """Menor t>0 com |rel + vel*t| = speed*t. rel/vel: (..., 2). inf se nao houver."""
    rel, vel = np.asarray(rel, float), np.asarray(vel, float)
    a = np.sum(vel * vel, -1) - speed**2
    b = 2.0 * np.sum(rel * vel, -1)
    c = np.sum(rel * rel, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tl = -c / b
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    tq = np.where(t1 > 0, t1, np.where(t2 > 0, t2, np.inf))
    tq = np.where(disc >= 0, tq, np.inf)
    return np.where(np.abs(a) < 1e-9, np.where(tl > 0, tl, np.inf), tq)


# ============================================================================ trajetorias
class PathFollower:
    """Posicao/velocidade ao longo da rota da missao. Sem loop, o fim da rota e o ponto de impacto."""

    def __init__(self, wps, speed: float, loop: bool, dive: float = IMPACT_DIVE):
        P = np.array([[w.n, w.e] for w in wps], float)
        self.dive = dive
        self.agl = np.array([max(w.agl, 20.0) for w in wps])
        if loop and len(P) > 2:
            P = np.vstack([P, P[:1]])
            self.agl = np.append(self.agl, self.agl[0])
        self.P, self.loop, self.speed = P, loop and len(P) > 2, speed
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(max(self.cum[-1], 1.0))
        self.impact_ne = P[-1].copy()
        if len(P) > 1:                                      # normais (esquerda) por segmento e nos vertices
            u = np.diff(P, axis=0) / np.maximum(seg, 1e-6)[:, None]
            sn = np.stack([-u[:, 1], u[:, 0]], 1)
            vn = np.vstack([sn[:1], sn[:-1] + sn[1:], sn[-1:]])
            bad = np.linalg.norm(vn, axis=1) < 1e-3                # meia-volta: usa a normal do segmento
            vn[bad] = np.vstack([sn[:1], sn])[bad]
            self.u, self.vn = u, vn / np.linalg.norm(vn, axis=1)[:, None]

    def crashed(self, s: float) -> bool:
        return (not self.loop) and s >= self.L

    def frame(self, s: float):
        """(posicao, tangente unitaria, normal unitaria continua, agl) apos s metros."""
        pos, v, agl = self.at(s)
        if len(self.P) < 2:
            return pos, np.array([1.0, 0.0]), np.array([0.0, 1.0]), agl
        s = s % self.L if self.loop else min(s, self.L)
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.cum) - 2))
        f = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-6)
        nrm = (1 - f) * self.vn[i] + f * self.vn[i + 1]
        return pos, self.u[i], nrm / max(np.linalg.norm(nrm), 1e-6), agl

    def at(self, s: float):
        """(posicao n,e, velocidade n,e, agl) apos percorrer s metros desde o inicio da rota."""
        s = s % self.L if self.loop else min(s, self.L)
        if len(self.P) < 2:
            return self.P[0].copy(), np.array([self.speed, 0.0]), float(self.agl[0])
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.cum) - 2))
        f = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-6)
        pos = self.P[i] + f * (self.P[i + 1] - self.P[i])
        d = self.P[i + 1] - self.P[i]
        v = d / max(np.linalg.norm(d), 1e-6) * self.speed
        agl = float(self.agl[i] * (1 - f) + self.agl[i + 1] * f)
        if not self.loop:                                   # rampa de mergulho ate o impacto
            agl = min(agl, self.dive * (self.L - s))
        return pos, v, agl


class _Body:
    """Aeronave minima compativel com RadarNetwork (alive, name, rcs, aircraft.x/alt/vel_ned)."""

    def __init__(self, name: str, rcs: float):
        self.name, self.rcs, self.alive = name, rcs, True
        self.aircraft = self
        self.x = np.zeros(13)
        self.alt = 0.0
        self.vel_ned = np.zeros(3)


# ============================================================================== cenarios
class Scenario:
    """Missao + terreno carregados uma vez (o DEM e a parte cara)."""

    def __init__(self, mission: Mission, terrain=None, synthetic: bool = False, name: str | None = None):
        mission.validate()
        self.m = mission
        self.name = name or mission.name
        if not mission.bases or sum(b.drones for b in mission.bases) == 0:
            raise ValueError("sem bases/drones (use o modo ⛽ Base no construtor)")
        if not mission.radars:
            raise ValueError("sem radares: os drones so conhecem as pistas de radar")
        self.terrain = terrain or load_terrain(mission.origin, synthetic)
        self.zones = np.array([[z.n, z.e, z.radius] for z in mission.zones], float).reshape(-1, 3)

    @classmethod
    def from_file(cls, path: str, synthetic: bool = False, root: str | None = None) -> "Scenario":
        """Nome do cenario = caminho relativo a `root` sem extensao (ex.: itajuba/1_radar/2_abastecimentos/a)."""
        name = os.path.splitext(os.path.relpath(path, root) if root else os.path.basename(path))[0]
        return cls(Mission.load(path), synthetic=synthetic, name=name.replace(os.sep, "/"))

    def zone_of(self, ne) -> int:
        """Indice da zona prioritaria que contem o ponto (-1 = fora de todas)."""
        if len(self.zones) == 0:
            return -1
        d = np.hypot(self.zones[:, 0] - ne[0], self.zones[:, 1] - ne[1]) - self.zones[:, 2]
        return int(np.argmin(d)) if d.min() <= 0 else -1

    def zone_geometry(self, ne, v):
        """(distancia assinada a fronteira da zona mais proxima [m], velocidade de aproximacao ao centro [m/s],
        dentro?). Sem zonas: (1e5, 0, False)."""
        if len(self.zones) == 0:
            return 1e5, 0.0, False
        rel = self.zones[:, :2] - np.asarray(ne)
        dist = np.linalg.norm(rel, axis=1)
        k = int(np.argmin(dist - self.zones[:, 2]))
        close = float(rel[k] @ np.asarray(v) / max(dist[k], 1.0))
        sd = float(dist[k] - self.zones[k, 2])
        return sd, close, sd <= 0


def load_scenarios(patterns, synthetic: bool = False, verbose: bool = True) -> list[Scenario]:
    """Carrega cenarios de arquivos/pastas/globs (ex.: 'missions'); ignora missoes sem bases ou radares."""
    paths: list[tuple[str, str | None]] = []
    for p in ([patterns] if isinstance(patterns, str) else patterns):
        if os.path.isdir(p):                                    # inclui subpastas (local/radares/abastecimentos)
            paths += [(f, p) for f in sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))]
        else:
            paths += [(f, None) for f in (sorted(glob.glob(p)) or [p])]
    out = []
    terrains: dict = {}                                         # um DEM por origem (varios cenarios no mesmo local)
    for p, root in paths:
        try:
            m = Mission.load(p)
            key = (m.origin["lat"], m.origin["lon"], m.origin["size_km"], synthetic)
            sc = Scenario.from_file(p, synthetic, root) if key not in terrains else None
            if sc is None:
                name = os.path.splitext(os.path.relpath(p, root) if root else os.path.basename(p))[0]
                sc = Scenario(m, terrains[key], name=name.replace(os.sep, "/"))
            terrains.setdefault(key, sc.terrain)
            out.append(sc)
        except (ValueError, KeyError) as ex:
            if verbose:
                print(f"  (ignorado {os.path.basename(p)}: {ex})")
    return out


# ================================================================================== ambiente
class RefuelEnv:
    skip_obs = False        # True: step() nao monta a observacao (politicas que leem o ambiente direto)

    def __init__(self, scenario: Scenario, cfg: RefuelConfig | None = None, seed: int = 0):
        self.sc, self.cfg = scenario, cfg or RefuelConfig()
        self.reset(seed)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None):
        c = self.cfg
        self.seed = 0 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        r = self.rng
        m = self.m = self.sc.m.randomized(r) if self.sc.m.is_preset else self.sc.m   # instancia sorteada do preset
        self.t = 0.0
        self.t_max = min(c.t_max, m.duration)
        self.rf = m.refuel
        self.bodies, self.paths, self.s, self.ev, self.ac_t0 = [], [], [], [], []
        for a in m.aircraft:
            wps = list(a.route_waypoints())
            if a.start:                                         # a rota comeca na posicao inicial
                wps = [Waypoint(a.start["n"], a.start["e"], agl=wps[0].agl)] + wps
            spd = float(np.clip((wps[-1].speed or a.speed), 10.0, 90.0)) * (
                1 + r.uniform(-c.speed_jitter, c.speed_jitter))
            pf = PathFollower(wps, spd, a.loop, a.dive or IMPACT_DIVE)
            lo, hi = c.start_frac
            self.paths.append(pf)
            self.s.append(r.uniform(lo, hi) * pf.L if pf.L > 1 and a.delay <= 0 else 0.0)
            self.bodies.append(_Body(a.name, a.rcs))
            self.ev.append(make_evasion(a.evasion, r))
            self.ac_t0.append(float(a.delay))
        self.n_ac = len(self.bodies)
        self.ac_t0 = np.array(self.ac_t0)
        self.ac_active = self.ac_t0 <= 0
        self.ac_pos = np.zeros((self.n_ac, 2))
        self.ac_vel = np.zeros((self.n_ac, 2))
        self.ac_acc = np.zeros((self.n_ac, 2))
        self.ac_agl = np.zeros(self.n_ac)
        self.ac_status = np.zeros(self.n_ac, int)               # 0 na lista, 1 reabastecida, 2 caiu
        self.ac_done_t = np.full(self.n_ac, np.nan)
        # penalidade por aeronave, conforme a zona do ponto de impacto
        self.ac_zone = np.array([self.sc.zone_of(p.impact_ne) for p in self.paths])
        self.ac_value = np.array([a.valor for a in m.aircraft], float)        # isca = 0
        self.ac_penalty = np.where(self.ac_zone >= 0, self.rf.crash_penalty_priority,
                                   self.rf.crash_penalty_other) * self.ac_value
        self._move_aircraft(0.0)
        self.radar_net = RadarNetwork(list(m.radars), self.sc.terrain, seed=self.seed)
        self.radar_ne = np.array([[rd.n, rd.e] for rd in m.radars], float).reshape(-1, 2)
        bases = [(b.n, b.e) for b in m.bases for _ in range(b.drones)]
        self.dr_pos = np.array(bases, float)
        self.n_dr = len(bases)
        self.dr_vel = np.zeros((self.n_dr, 2))
        self.dr_locked = np.zeros(self.n_dr, bool)              # chegou a fase terminal (sensor proprio)
        self.dr_state = np.zeros(self.n_dr, int)                # 0 ocioso, 1 em missao, 2 usado, 3 falhou
        self.dr_track = np.full(self.n_dr, -1)                  # id da pista perseguida
        self.dr_lock = np.full(self.n_dr, -1)                   # aeronave no sensor proprio (-1 = so radar)
        self.dr_next = np.zeros(self.n_dr)                      # proxima reavaliacao do alvo (drone em transito)
        self.dr_lost = np.zeros(self.n_dr, bool)                # pista perseguida sumiu do quadro de radar
        self.dr_fly = np.zeros(self.n_dr)
        self.dr_last = np.zeros((self.n_dr, 4))                 # ultima pista conhecida (n, e, vn, ve)
        self.dr_last_t = np.zeros(self.n_dr)
        self.targeted: set[int] = set()                         # pistas com pelo menos um drone
        self.tcount: dict[int, int] = {}                        # drones por pista
        self.info = {"refueled": 0, "failed": 0, "crashed": 0, "crashed_priority": 0, "retasks": 0, "misses": 0}
        self._pic, self._pic_by_id = [], {}
        self.queue: list[int] = []
        self.slots: list[dict | None] = [None] * c.max_tracks
        self._refresh_picture()
        self._advance_to_decision(advance_first=False)
        return self.obs()

    def _threat(self, k: int):
        """Drone em voo mais ameacador para a aeronave k (menor tempo ate o encontro): (rel, aproximacao)."""
        fly = np.where(self.dr_state == 1)[0]
        if not len(fly):
            return None, 0.0
        rel = self.dr_pos[fly] - self.ac_pos[k]
        d = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
        close = -np.sum(rel * (self.dr_vel[fly] - self.ac_vel[k]), 1) / d
        tgo = np.where(close > 1.0, d / np.maximum(close, 1e-6), np.inf)
        i = int(np.argmin(tgo))
        return (rel[i], float(close[i])) if np.isfinite(tgo[i]) else (None, 0.0)

    def _move_aircraft(self, h: float, t: float | None = None):
        """Avanca as aeronaves ativas h segundos (rota + manobra evasiva) e posiciona os corpos para o radar."""
        t = self.t if t is None else t
        for k in range(self.n_ac):
            if self.ac_status[k] != 0:
                continue
            if not self.ac_active[k]:
                if t + h < self.ac_t0[k]:
                    continue
                self.ac_active[k] = True
            pf, ev = self.paths[k], self.ev[k]
            v0 = self.ac_vel[k].copy()
            if ev is None:
                self.s[k] += pf.speed * h
                pos, v, agl = pf.at(self.s[k])
            else:
                if ev.react > 0 and h > 0:
                    _, _, nrm0, _ = pf.frame(self.s[k])
                    rel, close = self._threat(k)
                    ev.alert(t, rel, close, nrm0)
                y_lim = np.inf if pf.loop else 0.3 * max(pf.L - self.s[k], 0.0)
                y, yd, mult = ev.step(t, h, y_lim, pf.speed) if h > 0 else (ev.y, ev.yd, ev.mult)
                self.s[k] += pf.speed * mult * h
                pos, tan, nrm, agl = pf.frame(self.s[k])
                pos = pos + y * nrm
                v = tan * pf.speed * mult + nrm * yd
            self.ac_pos[k], self.ac_vel[k], self.ac_agl[k] = pos, v, agl
            self.ac_acc[k] = (v - v0) / h if h > 0 else 0.0
        hs = self.sc.terrain.height(self.ac_pos[:, 0], self.ac_pos[:, 1]) if self.n_ac else []
        for k, b in enumerate(self.bodies):
            b.x[0:2] = self.ac_pos[k]
            b.alt = float(hs[k]) + self.ac_agl[k]
            b.vel_ned = np.array([self.ac_vel[k][0], self.ac_vel[k][1], 0.0])

    def _crash_check(self) -> float:
        reward = 0.0
        for k in range(self.n_ac):
            if self.ac_status[k] == 0 and self.ac_active[k] and self.paths[k].crashed(self.s[k]):
                self.ac_status[k], self.ac_done_t[k] = 2, self.t
                self.info["crashed"] += 1
                self.info["crashed_priority"] += int(self.ac_zone[k] >= 0 and self.ac_value[k] > 0)
                reward -= float(self.ac_penalty[k])
        return reward

    def _live(self) -> np.ndarray:
        return np.where((self.ac_status == 0) & self.ac_active)[0]

    def _claim(self, tid: int):
        self.tcount[tid] = self.tcount.get(tid, 0) + 1
        self.targeted.add(tid)

    def _release(self, tid: int):
        n = self.tcount.get(tid, 0) - 1
        if n <= 0:
            self.tcount.pop(tid, None)
            self.targeted.discard(tid)
        else:
            self.tcount[tid] = n

    # ----------------------------------------------------------- informacao
    def _refresh_picture(self):
        """Quadro de radar (unica fonte de informacao dos drones). Pistas de aeronaves ja reabastecidas somem
        (aviso por data link); as que cairam deixam pista fantasma ate o rastreador descarta-la."""
        done = {self.bodies[k].name for k in range(self.n_ac) if self.ac_status[k] == 1}
        self._pic = [p for p in self.radar_net.picture(self.t) if p["truth"] not in done]
        self._pic_by_id = {p["id"]: p for p in self._pic}
        if self._pic:                                           # AGL de todas as pistas numa chamada so
            ne = np.array([[p["n"], p["e"]] for p in self._pic])
            for p, h in zip(self._pic, np.atleast_1d(self.sc.terrain.height(ne[:, 0], ne[:, 1]))):
                p["agl"] = float(p["alt"] - h)

    # ------------------------------------------------------------- dinamica
    def _track_memory(self, d: int):
        """Atualiza a ultima pista conhecida do drone d (e marca reavaliacao imediata se a pista acabou de sumir)."""
        tr = self._pic_by_id.get(self.dr_track[d])
        if tr is not None:
            self.dr_last[d] = (tr["n"], tr["e"], tr["vn"], tr["ve"])
            self.dr_last_t[d] = self.t
        elif not self.dr_lost[d]:                               # acabou de perder a pista: reavalia ja
            self.dr_next[d] = self.t
        self.dr_lost[d] = tr is None

    def _aim(self, d: int, t: float) -> np.ndarray:
        """Transito: ponto de interceptacao previsto pela ultima pista de radar conhecida."""
        pn, pe, vn, ve = self.dr_last[d]
        vel = np.array([vn, ve])
        est = np.array([pn, pe]) + vel * (t - self.dr_last_t[d])
        ti = intercept_time(est - self.dr_pos[d], vel, self.rf.drone_speed)
        return est + vel * (ti if np.isfinite(ti) else 0.0)

    def _contact(self, d: int, hit: int, t: float) -> float:
        """Contato (distancia VERDADEIRA): reabastece na hora."""
        self.ac_status[hit], self.ac_done_t[hit] = 1, t
        self.dr_state[d], self.dr_lock[d] = 2, -1
        self._release(int(self.dr_track[d]))
        self.info["refueled"] += 1
        if self.ac_value[hit] > 0:
            return REFUEL_REWARD
        return -self.rf.waste_penalty                                  # abater isca: interceptador desperdicado

    def _out_of_fuel(self, d: int) -> float:
        self.dr_state[d], self.dr_lock[d] = 3, -1
        self._release(int(self.dr_track[d]))
        self.info["failed"] += 1
        return -self.rf.waste_penalty                                  # lancado e nao usado

    def _step_world(self) -> float:
        reward = self._step_world_ideal() if self.cfg.guidance == "ideal" else self._step_world_dyn()
        return reward - float(np.sum(self.ac_status == 0)) * self.cfg.dt / (self.n_ac * 300.0)

    def _step_world_ideal(self) -> float:
        """Modelo original: o drone aponta instantaneamente para o alvo (sem limite de curva), passo de 1 s."""
        dt = self.cfg.dt
        self._move_aircraft(dt)
        reward = self._crash_check()                            # chegou ao impacto sem abastecer
        for k, b in enumerate(self.bodies):
            b.alive = bool(self.ac_status[k] == 0 and self.ac_active[k])
        self.radar_net.step(self.t, self.bodies)
        self.t += dt
        self._refresh_picture()
        vd = self.rf.drone_speed
        live = [int(k) for k in self._live()]
        for d in np.where(self.dr_state == 1)[0]:
            self._track_memory(d)
            # rendezvous, fase final: aeronave no alcance do sensor proprio -> segue a posicao verdadeira
            dist = [np.linalg.norm(self.ac_pos[k] - self.dr_pos[d]) for k in live]
            lock = live[int(np.argmin(dist))] if live and min(dist) <= self.rf.sensor_range else -1
            self.dr_lock[d] = lock
            self.dr_locked[d] |= lock >= 0
            # aeronaves ja andaram neste passo; transito: ponto de interceptacao da pista de radar
            aim = self.ac_pos[lock].copy() if lock >= 0 else self._aim(d, self.t)
            p0 = self.dr_pos[d].copy()
            dirv = aim - self.dr_pos[d]
            nrm = np.linalg.norm(dirv)
            if nrm > 1e-6:
                self.dr_pos[d] += dirv / nrm * min(vd * dt, nrm)
            self.dr_vel[d] = (self.dr_pos[d] - p0) / dt
            self.dr_fly[d] += dt
            hit = next((k for k in live if np.linalg.norm(self.ac_pos[k] - self.dr_pos[d]) <= self.rf.contact_radius),
                       None)
            if hit is not None:
                reward += self._contact(d, hit, self.t)
                live = [k for k in live if k != hit]
            elif self.dr_fly[d] >= self.rf.endurance:
                reward += self._out_of_fuel(d)
        return reward

    def _step_world_dyn(self) -> float:
        """Drones com velocidade constante e aceleracao lateral limitada; transito pela pista de radar e fase
        terminal pela lei de guiagem ``cfg.guidance`` com o sensor proprio. Perto das aeronaves o passo e
        subdividido (``terminal_dt``) e o contato e verificado pela distancia minima de passagem no subpasso."""
        c, dt, vd = self.cfg, self.cfg.dt, self.rf.drone_speed
        reward = 0.0
        fly = np.where(self.dr_state == 1)[0]
        for d in fly:
            self._track_memory(d)
        live = self._live()
        n_sub = 1
        if len(fly) and len(live):
            react = max((e.react for e in self.ev if e is not None), default=0.0)
            thr = max(self.rf.sensor_range, react) + (vd + 50.0) * dt
            D = np.linalg.norm(self.dr_pos[fly][:, None, :] - self.ac_pos[live][None, :, :], axis=2)
            if D.min() <= thr:
                n_sub = max(1, int(round(dt / c.terminal_dt)))
        h = dt / n_sub
        for i in range(n_sub):
            ts = self.t + i * h
            was = self.ac_active.copy()
            A0, V0, AC0 = self.ac_pos.copy(), self.ac_vel.copy(), self.ac_acc.copy()
            self._move_aircraft(h, ts)
            new = self.ac_active & ~was                          # entrou agora: sem posicao anterior
            A0[new], V0[new] = self.ac_pos[new], self.ac_vel[new]
            reward += self._crash_check()
            live = self._live()
            for d in np.where(self.dr_state == 1)[0]:
                p0 = self.dr_pos[d].copy()
                lock = -1
                if len(live):
                    dist = np.linalg.norm(A0[live] - p0, axis=1)
                    j = int(np.argmin(dist))
                    lock = int(live[j]) if dist[j] <= self.rf.sensor_range else -1
                self.dr_lock[d] = lock
                self.dr_locked[d] |= lock >= 0
                if lock >= 0:
                    self.dr_vel[d] = guidance_step(c.guidance, p0, self.dr_vel[d], A0[lock], V0[lock], AC0[lock],
                                                   c.drone_amax, h, c.pn_gain)
                else:
                    self.dr_vel[d] = turn_toward(self.dr_vel[d], self._aim(d, ts) - p0, c.drone_amax, h)
                self.dr_pos[d] = p0 + self.dr_vel[d] * h
                self.dr_fly[d] += h
                hit = None
                if len(live):
                    miss = closest_approach(A0[live] - p0, self.ac_pos[live] - self.dr_pos[d])
                    j = int(np.argmin(miss))
                    hit = int(live[j]) if miss[j] <= self.rf.contact_radius else None
                if hit is not None:
                    reward += self._contact(d, hit, ts + h)
                    live = live[live != hit]
                elif self.dr_fly[d] >= self.rf.endurance:
                    reward += self._out_of_fuel(d)
        for k, b in enumerate(self.bodies):
            b.alive = bool(self.ac_status[k] == 0 and self.ac_active[k])
        self.radar_net.step(self.t, self.bodies)
        self.t += dt
        self._refresh_picture()
        return reward

    def airborne(self, d: int) -> bool:
        return bool(self.dr_state[d] == 1)

    def _pending(self) -> list[int]:
        """Drones que decidem nesta rodada: em transito com reavaliacao vencida (primeiro) e ociosos na base."""
        fly = np.where((self.dr_state == 1) & (self.dr_lock < 0) & (self.dr_next <= self.t + 1e-9))[0]             if self.cfg.retask else np.zeros(0, int)
        return [int(d) for d in fly] + [int(d) for d in np.where(self.dr_state == 0)[0]]

    def _advance_to_decision(self, advance_first: bool = False) -> float:
        total = 0.0
        self.queue = [] if advance_first else self._pending()
        while not self.queue and not self.done:
            for _ in range(int(round(self.cfg.decision_dt / self.cfg.dt))):
                total += self._step_world()
            self.queue = self._pending()
        self._build_slots()
        return total

    @property
    def done(self) -> bool:
        no_drones = not np.any((self.dr_state == 0) | (self.dr_state == 1))
        return bool(np.all(self.ac_status != 0) or self.t >= self.t_max or no_drones)

    @property
    def current_drone(self) -> int:
        return self.queue[0]

    # ------------------------------------------------------------ observacao
    def free_tracks(self, d: int | None = None) -> list[dict]:
        """Pistas nao designadas a outros drones (a pista do proprio drone `d` conta como livre)."""
        own = int(self.dr_track[d]) if d is not None and self.airborne(d) else None
        cap = self.cfg.max_per_track
        return [p for p in self._pic if self.tcount.get(p["id"], 0) < cap or p["id"] == own]

    def _agl(self, p: dict) -> float:
        a = p.get("agl")
        return a if a is not None else float(p["alt"] - self.sc.terrain.height(p["n"], p["e"]))

    def legit_feats(self, tracks: list[dict]) -> np.ndarray:
        """Caracteristicas so de radar (independentes do drone) para decidir se a pista e legitima."""
        f = np.zeros((len(tracks), LEGIT_FEATS), np.float32)
        for i, p in enumerate(tracks):
            sp = float(np.hypot(p["vn"], p["ve"]))
            dr = float(np.min(np.hypot(*(self.radar_ne - [p["n"], p["e"]]).T))) if len(self.radar_ne) else 1e4
            f[i] = (min(sp, 180.0) / 30.0, np.clip(p["vh"], -30.0, 30.0) / 5.0, min(abs(p["vh"]), 30.0) / 5.0,
                    np.clip(self._agl(p), -300.0, 900.0) / 300.0, min(p["sigma"], 400.0) / 200.0,
                    min(p["age"], 40.0) / 20.0, min(p["life"], 300.0) / 60.0, min(p["hits"], 30.0) / 10.0,
                    min(p["hits"] / max(p["life"], 1.0), 1.0) * 5.0, min(dr, 15000.0) / 5000.0)
        return f

    def plausible(self, p: dict) -> bool:
        """Regra fisica: velocidade, razao de subida e AGL dentro do envelope de uma aeronave."""
        c = self.rf.envelope or PLAUSIBLE
        sp, agl = float(np.hypot(p["vn"], p["ve"])), self._agl(p)
        return bool(c["v_min"] <= sp <= c["v_max"] and abs(p["vh"]) <= c["vh_max"] and c["agl_min"] <= agl <= c["agl_max"])

    def _build_slots(self):
        """Pistas livres (nao designadas), ordenadas pelo tempo de interceptacao do drone atual."""
        K = self.cfg.max_tracks
        self.slots = [None] * K
        if self.done or not self.queue:
            return
        d = self.current_drone
        tr = self.free_tracks(d)
        for p in tr:
            p["rel"] = np.array([p["n"], p["e"]]) - self.dr_pos[d]
            p["tint"] = float(intercept_time(p["rel"], np.array([p["vn"], p["ve"]]), self.rf.drone_speed))
        tr.sort(key=lambda p: (p["tint"] if np.isfinite(p["tint"]) else 1e9, p["id"]))
        for i, p in enumerate(tr[:K]):
            self.slots[i] = p

    def mask(self) -> np.ndarray:
        K = self.cfg.max_tracks
        m = np.zeros(K + 1, bool)
        m[:K] = [s is not None for s in self.slots]
        m[K] = True
        return m

    def obs(self) -> dict:
        c = self.cfg
        K = c.max_tracks
        f = np.zeros((K, TRACK_FEATS), np.float32)
        g = np.zeros(GLOB_FEATS, np.float32)
        lf = np.zeros((K, LEGIT_FEATS), np.float32)
        y = np.zeros(K, np.float32)
        if self.done or not self.queue:
            return {"ac": f, "lf": lf, "glob": g, "mask": np.ones(K + 1, bool), "legit": y}
        d = self.current_drone
        ext = self.sc.terrain.extent
        L = float(max(ext[1] - ext[0], ext[3] - ext[2]))
        T = self.t_max
        own = int(self.dr_track[d]) if self.airborne(d) else None
        tgt = [np.array([self._pic_by_id[i]["n"], self._pic_by_id[i]["e"]]) for i in self.targeted
               if i in self._pic_by_id and i != own]
        left = self.rf.endurance - self.dr_fly[d]
        for i, p in enumerate(self.slots):
            if p is None:
                continue
            v = np.array([p["vn"], p["ve"]])
            rel, ti = p["rel"], p["tint"]
            pos = np.array([p["n"], p["e"]])
            f[i, 0:2] = rel / L
            f[i, 2:4] = v / 30.0
            f[i, 4] = np.linalg.norm(rel) / L
            f[i, 5] = min(ti, 2 * T) / T if np.isfinite(ti) else 2.0
            f[i, 6] = float(np.isfinite(ti) and ti <= left)
            f[i, 7] = min(p["age"], 40.0) / 20.0
            f[i, 8] = min(p["sigma"], 400.0) / 200.0
            f[i, 9] = float(min((np.linalg.norm(pos - q) for q in tgt), default=1e5) < c.cover_radius)
            f[i, 10] = min(self.t - self.radar_net.track_log[p["id"]][0].t, T) / T
            agl = self._agl(p)                                                     # altitude e descida: so do radar
            f[i, 11] = np.clip(agl, -100.0, 600.0) / 300.0
            f[i, 12] = np.clip(p["vh"], -15.0, 15.0) / 5.0
            sd, close, inside = self.sc.zone_geometry(pos, v)                     # zonas prioritarias (comando)
            f[i, 13] = np.clip(sd, -L, L) / L
            f[i, 14] = np.clip(close, -40.0, 40.0) / 30.0
            f[i, 15] = float(inside)
            f[i, 16] = float(p["id"] == own)                                       # alvo atual deste drone
        g[0] = len(self._pending()) / max(self.n_dr, 1)
        g[1] = np.sum([s is not None for s in self.slots]) / K
        g[2] = self.t / T
        g[3] = left / self.rf.endurance
        g[4] = np.sum(self.dr_state <= 1) / max(self.n_dr, 1)
        g[5] = float(self.airborne(d))
        g[6] = min(self.t - self.dr_last_t[d], 60.0) / 30.0 if self.airborne(d) else 0.0   # idade da pista perseguida
        valid = [i for i, p in enumerate(self.slots) if p is not None]
        if valid:
            lf[valid] = self.legit_feats([self.slots[i] for i in valid])
            y[valid] = [self.slots[i]["truth"] is not None for i in valid]       # rotulo: so treino/metricas
        return {"ac": f, "lf": lf, "glob": g, "mask": self.mask(), "legit": y}

    # ------------------------------------------------------------------ passo
    def step(self, action: int):
        """action: slot da pista (0..K-1) ou K = esperar (na base) / manter a pista (em voo).
        Devolve (obs, recompensa, terminou)."""
        K = self.cfg.max_tracks
        d = self.queue.pop(0)
        if self.airborne(d):
            self.dr_next[d] = self.t + self.cfg.retask_dt
            p = self.slots[action] if action < K else None
            if p is not None and p["id"] != self.dr_track[d]:      # abandona a pista atual e assume outra
                self._release(int(self.dr_track[d]))
                self.dr_track[d], self.dr_lost[d] = p["id"], False
                self.dr_last[d] = (p["n"], p["e"], p["vn"], p["ve"])
                self.dr_last_t[d] = self.t
                self._claim(p["id"])
                self.info["retasks"] += 1
        elif action < K and self.slots[action] is not None:
            p = self.slots[action]
            self.dr_state[d], self.dr_track[d], self.dr_fly[d] = 1, p["id"], 0.0
            self.dr_next[d], self.dr_lost[d] = self.t + self.cfg.retask_dt, False
            self.dr_last[d] = (p["n"], p["e"], p["vn"], p["ve"])
            self.dr_last_t[d] = self.t
            self._claim(p["id"])
            aim = np.array([p["n"], p["e"]]) - self.dr_pos[d]           # lancado apontando para a pista
            self.dr_vel[d] = aim / max(np.linalg.norm(aim), 1e-6) * self.rf.drone_speed
        reward = 0.0
        if not self.queue:
            reward = self._advance_to_decision(advance_first=True)
        else:
            self._build_slots()
        reward += self._settle()
        if self.done and not getattr(self, "_waste_charged", False):     # em voo sem uso no fim: desperdicados
            self._waste_charged = True
            reward -= self.rf.waste_penalty * float(np.sum(self.dr_state == 1))
        return (None if self.skip_obs else self.obs()), reward, self.done

    def _settle(self) -> float:
        """Sem drones na base nem em voo, ninguem mais intercepta: as aeronaves restantes (inclusive as que ainda
        nao entraram) chegam ao ponto de impacto e contam como queda. Sem isso o episodio terminaria com aeronaves sem
        desfecho e o dano seria subestimado justamente quando a defesa satura."""
        if np.any((self.dr_state == 0) | (self.dr_state == 1)):
            return 0.0
        reward = 0.0
        for k in range(self.n_ac):
            if self.ac_status[k] == 0 and not self.paths[k].loop:
                eta = max(self.ac_t0[k] - self.t, 0.0) + max(self.paths[k].L - self.s[k], 0.0) / self.paths[k].speed
                self.ac_status[k], self.ac_done_t[k] = 2, self.t + eta
                self.info["crashed"] += 1
                self.info["crashed_priority"] += int(self.ac_zone[k] >= 0 and self.ac_value[k] > 0)
                reward -= float(self.ac_penalty[k])
        return reward

    def summary(self) -> dict:
        ok = self.ac_status == 1
        crashed = self.ac_status == 2
        real = self.ac_value > 0                                 # ameacas de verdade (sem iscas)
        # interceptadores desperdicados: sem autonomia, gastos em iscas, e (no fim) ainda em voo sem uso
        waste = int((self.dr_state == 3).sum() + (ok & ~real).sum() + (self.dr_state == 1).sum() * self.done)
        return {"refueled": int(ok.sum()), "crashed": int((crashed & real).sum()),
                "crashed_priority": int((crashed & real & (self.ac_zone >= 0)).sum()),
                "n_aircraft": self.n_ac, "rate": float(ok[real].mean()) if real.any() else float(ok.mean()),
                "n_iscas": int((~real).sum()), "iscas_abatidas": int((ok & ~real).sum()),
                "penalty": float(self.ac_penalty[crashed].sum()), "success": float(ok[real].all()),
                "mean_t": float(self.ac_done_t[ok].mean()) if ok.any() else float("nan"),   # tempo ate reabastecer
                "drones_failed": int((self.dr_state == 3).sum()), "retasks": self.info["retasks"],
                "dano": float(self.ac_penalty[crashed].sum() / max(self.ac_penalty.sum(), 1e-9)),  # dano normalizado
                "drones_usados": int((self.dr_state >= 1).sum()), "terminal": int(self.dr_locked.sum()),
                "desperdicio": waste, "custo": float((self.ac_penalty[crashed].sum() + self.rf.waste_penalty * waste)
                                                     / max(self.ac_penalty.sum(), 1e-9)),
                "evasoes": int(sum(e.breaks for e in self.ev if e is not None))}
