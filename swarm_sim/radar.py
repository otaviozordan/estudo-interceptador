"""Radares e rastreamento: a unica fonte de informacao sobre onde estao as aeronaves.

Modelo do radar (``RadarStation`` / ``RadarNetwork``)
  * Equacao do radar: SNR(r) = SNR50 * (R0/r)^4 * RCS_efetiva * perdas, em que R0 (``range_m``) e o alcance de
    deteccao (Pd = 50 %) para um alvo de 1 m2 em visada livre. Alvo flutuante Swerling I:
    Pd = Pfa^(1/(1+SNR)). SNR50 = ln(Pfa)/ln(0.5) - 1 (~13 dB para Pfa = 1e-6).
  * RCS por aeronave (``AircraftSpec.rcs``) com dependencia do aspecto: menor de frente/de cauda, maior de lado.
  * Perdas: atmosfera (dB/km) e difracao em gume de faca no relevo (Fresnel, depende da frequencia): a sombra do
    terreno atenua o sinal de forma gradual em vez do liga/desliga binario. Curvatura da Terra (raio efetivo 4/3).
  * Varredura: o feixe gira (ou varre o setor) com ``scan_period``; cada alvo e medido no instante em que o feixe
    passa por ele (nao ha medida instantanea de todos os alvos). Radares sao assincronos.
  * Ruido: sigma de distancia/azimute/elevacao nominal em SNR de referencia, escalando com 1/sqrt(SNR)
    (limite de Cramer-Rao); vieses sistematicos de distancia e azimute (desalinhamento) por radar, que o rastreador
    NAO conhece. Falsos alarmes concentrados perto do radar (clutter), nao uniformes no espaco.
  * Nao modelado: Doppler/clutter de solo detalhado, multipercurso, feixe com abertura em elevacao, latencia.

Rastreador (``Tracker``)
  * Fusao multi-radar com IMM (2 modelos de velocidade constante: ruido de aceleracao baixo/alto, para cobrir
    voo reto e manobra) sobre medidas convertidas para cartesiano (covariancia pelo Jacobiano de r/az/el).
  * Associacao por Mahalanobis + algoritmo hungaro, confirmacao por numero de plots, descarte por tempo sem
    atualizacao e fusao de pistas duplicadas.

O tracker NUNCA usa a identidade verdadeira das aeronaves; ela so e usada para avaliar o erro (``summary``).
Quem precisar saber onde as aeronaves estao (ex.: interceptadores) deve usar ``RadarNetwork.picture(t)``.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import linear_sum_assignment

from .terrain import Terrain

R_EARTH_EFF = 6_371_000.0 * 4.0 / 3.0       # raio efetivo (refracao padrao)
C_LIGHT = 299_792_458.0
SNR_REF = 20.0                               # SNR (linear) em que os sigmas nominais valem
MAX_DIFF_LOSS_DB = 60.0


# ======================================================================== radar
@dataclass
class RadarStation:
    n: float
    e: float
    name: str = "radar"
    height: float = 15.0            # altura da antena acima do solo [m]
    range_m: float = 40_000.0       # alcance para RCS = 1 m2, visada livre, Pd = 50 % [m]
    min_range: float = 300.0        # zona cega proxima [m]
    scan_period: float = 4.0        # periodo de varredura [s]
    range_sigma: float = 25.0       # sigma de distancia em SNR de referencia [m]
    az_sigma_deg: float = 0.3       # sigma de azimute em SNR de referencia [graus]
    el_sigma_deg: float = 0.6       # sigma de elevacao em SNR de referencia [graus]
    min_elev_deg: float = 0.0       # elevacao minima do feixe [graus]
    max_elev_deg: float = 70.0      # elevacao maxima (cone de silencio acima) [graus]
    az_center_deg: float = 0.0      # centro do setor [graus, 0 = norte, horario]
    az_width_deg: float = 360.0     # abertura do setor [graus] (360 = varredura completa)
    false_alarms: float = 0.0       # falsos alarmes medios por varredura
    frequency_ghz: float = 3.0      # frequencia (comprimento de onda da difracao) [GHz]
    atm_loss_db_km: float = 0.02    # atenuacao atmosferica (ida e volta) [dB/km]
    range_bias: float = 0.0         # vies sistematico de distancia [m]
    az_bias_deg: float = 0.0        # vies sistematico de azimute (desalinhamento) [graus]
    pfa: float = 1e-6               # probabilidade de falso alarme por celula
    enabled: bool = True

    # ---------------------------------------------------------------- geometria
    def antenna_alt(self, terrain: Terrain) -> float:
        return float(terrain.height(self.n, self.e)) + self.height

    @property
    def wavelength(self) -> float:
        return C_LIGHT / (self.frequency_ghz * 1e9)

    @property
    def az_start(self) -> float:
        return self.az_center_deg - min(self.az_width_deg, 360.0) / 2.0

    def sweep_phase(self, az_deg):
        """Fase [0,1] do azimute dentro da varredura (NaN fora do setor)."""
        w = min(self.az_width_deg, 360.0)
        d = (np.asarray(az_deg, float) - self.az_start) % 360.0
        return np.where(d <= w, d / w, np.nan) if w < 360.0 else d / 360.0

    def in_sector(self, az_deg: float) -> bool:
        return bool(np.isfinite(self.sweep_phase(az_deg)))

    # ------------------------------------------------------------ equacao do radar
    @property
    def snr50(self) -> float:
        return float(np.log(self.pfa) / np.log(0.5) - 1.0)

    def snr(self, r, rcs, loss_db=0.0):
        """SNR linear pela equacao do radar (r em m, rcs efetiva em m2, perda extra em dB)."""
        r = np.maximum(np.asarray(r, float), 1.0)
        tot_db = self.atm_loss_db_km * r / 1000.0 + loss_db
        return self.snr50 * (self.range_m / r) ** 4 * rcs * 10.0 ** (-tot_db / 10.0)

    def pd_from_snr(self, snr):
        return self.pfa ** (1.0 / (1.0 + np.maximum(snr, 0.0)))

    def pd(self, r, rcs=1.0, loss_db=0.0):
        return self.pd_from_snr(self.snr(r, rcs, loss_db))

    def sigmas(self, snr: float = SNR_REF) -> np.ndarray:
        s = float(np.clip(np.sqrt(SNR_REF / max(snr, 1e-9)), 0.5, 4.0))
        return s * np.array([self.range_sigma, np.radians(self.az_sigma_deg), np.radians(self.el_sigma_deg)])


_RADAR_FIELDS = {f.name for f in fields(RadarStation)}


def radar_from_dict(d: dict) -> RadarStation:
    return RadarStation(**{k: v for k, v in d.items() if k in _RADAR_FIELDS})


def aspect_factor(los_az_rad: float, heading_rad: float) -> float:
    """Fator de RCS pelo aspecto: 0.3 de frente/cauda, 1.7 de lado (media 1.0 sobre o aspecto)."""
    return 0.3 + 1.4 * np.sin(los_az_rad - heading_rad) ** 2


# ======================================================================= relevo
def _max_fresnel_v(terrain: Terrain, radar_pos, targets, K: int, wavelength: float) -> np.ndarray:
    """Maior parametro de difracao v (Fresnel-Kirchhoff) ao longo de cada visada radar -> alvo.

    targets: (N, 3) [n, e, alt_msl]. v > 0: obstaculo acima da visada; v < -0.78: livre de Fresnel."""
    targets = np.atleast_2d(np.asarray(targets, float))
    p0 = np.asarray(radar_pos, float)
    D = np.hypot(targets[:, 0] - p0[0], targets[:, 1] - p0[1])
    s = np.linspace(0.0, 1.0, K + 2)[1:-1]
    n = p0[0] + s[None, :] * (targets[:, 0] - p0[0])[:, None]
    e = p0[1] + s[None, :] * (targets[:, 1] - p0[1])[:, None]
    a = p0[2] + s[None, :] * (targets[:, 2] - p0[2])[:, None]
    d1 = s[None, :] * D[:, None]
    d2 = (1.0 - s[None, :]) * D[:, None]
    bulge = d1 * d2 / (2.0 * R_EARTH_EFF)
    h_obst = terrain.height(n, e) + bulge - a                     # > 0: terreno acima da visada
    dd = np.maximum(d1 * d2, 1.0)
    v = h_obst * np.sqrt(2.0 * np.maximum(d1 + d2, 1.0) / (wavelength * dd))
    v = np.where(d1 < 50.0, -np.inf, v)                           # ignora o terreno colado na antena
    return v.max(axis=1)


def diffraction_loss_db(v):
    """Perda por difracao em gume de faca [dB] (aprox. ITU-R P.526), 0 para v <= -0.78."""
    v = np.asarray(v, float)
    x = v - 0.1
    j = 6.9 + 20.0 * np.log10(np.sqrt(x * x + 1.0) + x)
    return np.clip(np.where(v > -0.78, j, 0.0), 0.0, MAX_DIFF_LOSS_DB)


def path_loss_db(terrain: Terrain, radar_pos, target, wavelength: float) -> float:
    """Perda de difracao do relevo entre o radar e um alvo [dB] (0 = visada livre com folga de Fresnel)."""
    D = float(np.hypot(target[0] - radar_pos[0], target[1] - radar_pos[1]))
    K = int(min(max(D / terrain.res, 8), 1500))
    return float(diffraction_loss_db(_max_fresnel_v(terrain, radar_pos, np.asarray(target)[None, :], K,
                                                    wavelength))[0])


def line_of_sight(terrain: Terrain, p0, p1, margin: float = 0.0) -> bool:
    """True se a visada p0 -> p1 (n, e, alt_msl) esta livre de terreno, com abaulamento da Terra."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    D = float(np.hypot(*(p1[:2] - p0[:2])))
    if D < 1.0:
        return True
    k = int(min(max(np.ceil(D / terrain.res), 2), 4000)) + 1
    s = np.linspace(0.0, 1.0, k)[1:-1]
    n = p0[0] + s * (p1[0] - p0[0])
    e = p0[1] + s * (p1[1] - p0[1])
    a = p0[2] + s * (p1[2] - p0[2])
    bulge = (s * D) * ((1.0 - s) * D) / (2.0 * R_EARTH_EFF)
    return bool(np.all(a > terrain.height(n, e) + bulge + margin))


def coverage_mask(terrain: Terrain, radar: RadarStation, agl: float, rcs: float = 1.0, cells: int = 90):
    """Cobertura (Pd >= 50 %) para um alvo a ``agl`` m do solo e RCS ``rcs``, incluindo difracao do relevo.

    Retorna (n_eixo, e_eixo, mascara[ni, ej])."""
    n_lo, n_hi, e_lo, e_hi = terrain.extent
    ns = np.linspace(n_lo, n_hi, cells)
    es = np.linspace(e_lo, e_hi, cells)
    N, E = np.meshgrid(ns, es, indexing="ij")
    alt_t = terrain.height(N, E) + agl
    ra = radar.antenna_alt(terrain)
    dn, de = N - radar.n, E - radar.e
    D = np.hypot(dn, de)
    slant = np.hypot(D, alt_t - ra)
    az = np.degrees(np.arctan2(de, dn)) % 360.0
    el = np.degrees(np.arctan2(alt_t - ra - D * D / (2 * R_EARTH_EFF), np.maximum(D, 1e-6)))
    ok = (slant >= radar.min_range) & (el >= radar.min_elev_deg) & (el <= radar.max_elev_deg)
    ok &= np.isfinite(radar.sweep_phase(az))
    ok &= radar.snr(slant, rcs) >= radar.snr50                      # alcance em espaco livre
    idx = np.flatnonzero(ok.ravel())
    if idx.size:
        tg = np.stack([N.ravel()[idx], E.ravel()[idx], alt_t.ravel()[idx]], axis=1)
        v = _max_fresnel_v(terrain, (radar.n, radar.e, ra), tg, 96, radar.wavelength)
        loss = diffraction_loss_db(v)
        good = radar.snr(slant.ravel()[idx], rcs, loss) >= radar.snr50
        flat = ok.ravel().copy()
        flat[idx] = good
        ok = flat.reshape(ok.shape)
    return ns, es, ok


# ===================================================================== medidas
def polar_to_ned(radar_pos, r, az, el):
    rn, re, ra = radar_pos
    ce = np.cos(el)
    return np.array([rn + r * ce * np.cos(az), re + r * ce * np.sin(az), ra + r * np.sin(el)])


def measurement_cov(r, az, el, sig) -> np.ndarray:
    """Covariancia cartesiana (n, e, h) de uma medida (r, az, el) com desvios ``sig``."""
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    J = np.array([[ce * ca, -r * ce * sa, -r * se * ca],
                  [ce * sa, r * ce * ca, -r * se * sa],
                  [se, 0.0, r * ce]])
    return J @ np.diag(np.square(sig)) @ J.T


@dataclass
class Detection:
    t: float
    radar: int
    z: np.ndarray          # (n, e, alt) medido
    R: np.ndarray          # 3x3
    truth: str | None      # so para avaliacao (None = falso alarme)


# ===================================================================== tracker
def _gauss_lik(y, S) -> float:
    d = float(y @ np.linalg.solve(S, y))
    return float(np.exp(-0.5 * d) / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(S)))


class Track:
    """Pista com IMM de dois modelos de velocidade constante (ruido de aceleracao baixo / alto)."""
    Q_ACCEL = (1.0, 8.0)                  # sigma de aceleracao dos modelos [m/s2]
    P_STAY = 0.97

    def __init__(self, tid: int, t: float, det: Detection, vsig=(40.0, 40.0, 15.0)):
        self.id, self.t, self.hits = tid, t, 1
        self.t_born = t
        x = np.concatenate([det.z, np.zeros(3)])
        P = np.zeros((6, 6))
        P[:3, :3] = det.R
        P[3:, 3:] = np.diag(np.square(vsig))
        self.xs = [x.copy(), x.copy()]
        self.Ps = [P.copy(), P.copy()]
        self.mu = np.array([0.8, 0.2])
        self.Pi = np.array([[self.P_STAY, 1 - self.P_STAY], [1 - self.P_STAY, self.P_STAY]])
        self.confirmed = False
        self.t_upd = t
        self._combine()

    def _combine(self):
        self.x = sum(m * x for m, x in zip(self.mu, self.xs))
        self.P = sum(m * (P + np.outer(x - self.x, x - self.x)) for m, x, P in zip(self.mu, self.xs, self.Ps))

    @staticmethod
    def _F_Q(dt: float, q: float):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        qn = np.array([q, q, 0.5 * q]) ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = qn[i] * dt ** 4 / 4
            Q[i, i + 3] = Q[i + 3, i] = qn[i] * dt ** 3 / 2
            Q[i + 3, i + 3] = qn[i] * dt ** 2
        return F, Q

    def predict(self, t: float, q: float = 0.0):
        dt = t - self.t
        if dt <= 0:
            return
        c = self.Pi.T @ self.mu                                  # prob. predita dos modelos
        mix = self.Pi * self.mu[:, None] / c[None, :]            # mix[i, j] = P(modelo i | modelo j)
        x0, P0 = [], []
        for j in range(2):
            xj = sum(mix[i, j] * self.xs[i] for i in range(2))
            Pj = sum(mix[i, j] * (self.Ps[i] + np.outer(self.xs[i] - xj, self.xs[i] - xj)) for i in range(2))
            x0.append(xj)
            P0.append(Pj)
        for j in range(2):
            F, Q = self._F_Q(dt, self.Q_ACCEL[j])
            self.xs[j] = F @ x0[j]
            self.Ps[j] = F @ P0[j] @ F.T + Q
        self.mu = c
        self.t = t
        self._combine()

    def innovation(self, z, R):
        return z - self.x[:3], self.P[:3, :3] + R

    def update(self, z, R):
        lik = np.zeros(2)
        for j in range(2):
            y = z - self.xs[j][:3]
            S = self.Ps[j][:3, :3] + R
            K = self.Ps[j][:, :3] @ np.linalg.inv(S)
            lik[j] = _gauss_lik(y, S)
            self.xs[j] = self.xs[j] + K @ y
            I_KH = np.eye(6)
            I_KH[:, :3] -= K
            self.Ps[j] = I_KH @ self.Ps[j] @ I_KH.T + K @ R @ K.T      # forma de Joseph
        w = self.mu * np.maximum(lik, 1e-300)
        self.mu = w / w.sum() if w.sum() > 0 else self.mu
        self.hits += 1
        self.t_upd = self.t
        self._combine()


class Tracker:
    def __init__(self, gate: float = 20.0, confirm_hits: int = 3, drop_after: float = 20.0,
                 tentative_drop: float = 10.0):
        self.gate, self.confirm_hits = gate, confirm_hits
        self.drop_after, self.tentative_drop = drop_after, tentative_drop
        self.tracks: list[Track] = []
        self._next = 1

    def process(self, t: float, dets: list[Detection]) -> list[Track]:
        """Processa as deteccoes de um instante/radar; devolve as pistas atualizadas."""
        self._prune(t)
        for tr in self.tracks:
            tr.predict(t)
        updated: list[Track] = []
        used = set()
        if self.tracks and dets:
            cost = np.full((len(self.tracks), len(dets)), 1e9)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    y, S = tr.innovation(d.z, d.R)
                    m2 = float(y @ np.linalg.solve(S, y))
                    if m2 < self.gate:
                        cost[i, j] = m2
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e8:
                    tr = self.tracks[i]
                    tr.update(dets[j].z, dets[j].R)
                    if not tr.confirmed and tr.hits >= self.confirm_hits:
                        tr.confirmed = True
                    updated.append(tr)
                    used.add(j)
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append(Track(self._next, t, d))
                self._next += 1
        self._merge_duplicates()
        return [tr for tr in updated if tr in self.tracks]

    def _prune(self, t: float):
        self.tracks = [tr for tr in self.tracks
                       if t - tr.t_upd <= (self.drop_after if tr.confirmed else self.tentative_drop)]

    def _merge_duplicates(self):
        """Mesmo alvo visto por dois radares que nao casaram no portao: fica a pista mais antiga."""
        conf = sorted((tr for tr in self.tracks if tr.confirmed), key=lambda r: -r.hits)
        drop = set()
        for i, a in enumerate(conf):
            if a.id in drop:
                continue
            for b in conf[i + 1:]:
                if b.id not in drop and np.linalg.norm(a.x[:3] - b.x[:3]) < 150.0 and \
                        np.linalg.norm(a.x[3:] - b.x[3:]) < 15.0:
                    drop.add(b.id)
        if drop:
            self.tracks = [tr for tr in self.tracks if tr.id not in drop]


# ====================================================================== rede
@dataclass
class TrackSample:
    t: float
    x: np.ndarray            # estado [n, e, h, vn, ve, vh]
    sigma: float             # incerteza de posicao (raiz do traco de P pos / 3) [m]
    confirmed: bool
    truth: str | None        # aeronave mais proxima (avaliacao)
    err: float               # erro de posicao 3D em relacao a ``truth`` [m]
    hits: int = 0            # medidas associadas a pista ate t
    born: float = 0.0        # instante de criacao da pista


class RadarNetwork:
    def __init__(self, radars: list[RadarStation], terrain: Terrain, seed: int = 0,
                 tracker: Tracker | None = None):
        self.radars = list(radars)
        self.terrain = terrain
        self.tracker = tracker or Tracker()
        self.rng = np.random.default_rng(seed + 1234)
        self.phase0 = [float(self.rng.random()) for _ in self.radars]      # fase inicial do feixe
        self._u_prev: list[float | None] = [None] * len(self.radars)
        self._sweep = [[0, 0] for _ in self.radars]                         # [plots, visiveis] da varredura atual
        self.detections: list[Detection] = []
        self.track_log: dict[int, list[TrackSample]] = {}
        self.scan_log: list[tuple[float, int, int, int]] = []      # (t, radar, plots, visiveis) por varredura completa
        self._ant = [r.antenna_alt(terrain) for r in self.radars]

    # ------------------------------------------------------------- geometria / enlace
    def link(self, k: int, target, heading_rad: float, rcs: float = 1.0) -> dict:
        """Geometria e orcamento de enlace radar k -> alvo (n, e, alt): distancia, az, el, perda, SNR, Pd."""
        rd = self.radars[k]
        pos = (rd.n, rd.e, self._ant[k])
        dn, de, dh = target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]
        hd = float(np.hypot(dn, de))
        r = float(np.sqrt(hd * hd + dh * dh))
        az = float(np.degrees(np.arctan2(de, dn)) % 360.0)
        el = float(np.degrees(np.arctan2(dh - hd * hd / (2 * R_EARTH_EFF), max(hd, 1e-6))))
        rcs_eff = rcs * aspect_factor(np.radians(az), heading_rad)
        loss = path_loss_db(self.terrain, pos, np.asarray(target, float), rd.wavelength)
        snr = float(rd.snr(r, rcs_eff, loss))
        return {"r": r, "az": az, "el": el, "loss_db": loss, "rcs_eff": rcs_eff, "snr": snr,
                "pd": float(rd.pd_from_snr(snr)),
                "in_beam": rd.min_elev_deg <= el <= rd.max_elev_deg and rd.min_range <= r and rd.in_sector(az)}

    # ------------------------------------------------------------- varredura
    def step(self, t: float, agents):
        for k, rd in enumerate(self.radars):
            if not rd.enabled:
                continue
            u = ((t / rd.scan_period) + self.phase0[k]) % 1.0              # fase do feixe na varredura [0,1)
            u0, self._u_prev[k] = self._u_prev[k], u
            if u0 is None:
                continue
            wrap = u < u0
            dets = self._beam_pass(t, k, rd, agents, u0, u, wrap)
            dets += self._false_alarms(t, k, rd, u0, u, wrap)
            if wrap:
                self.scan_log.append((t, k, *self._sweep[k]))
                self._sweep[k] = [0, 0]
            if dets:
                self._sweep[k][0] += len(dets)
                self.detections.extend(dets)
                for tr in self.tracker.process(t, dets):
                    self._log_track(t, tr, agents)

    @staticmethod
    def _crossed(pt: float, u0: float, u: float, wrap: bool) -> bool:
        return (pt > u0 or pt <= u) if wrap else (u0 < pt <= u)

    def _beam_pass(self, t, k, rd, agents, u0, u, wrap) -> list[Detection]:
        pos = (rd.n, rd.e, self._ant[k])
        dets: list[Detection] = []
        for a in agents:
            if not a.alive:
                continue
            ac = a.aircraft
            dn, de = ac.x[0] - rd.n, ac.x[1] - rd.e
            az = float(np.degrees(np.arctan2(de, dn)) % 360.0)
            pt = float(rd.sweep_phase(az))
            if not np.isfinite(pt) or not self._crossed(pt, u0, u, wrap):
                continue                                      # o feixe nao passou por este alvo agora
            target = np.array([ac.x[0], ac.x[1], ac.alt])
            vn = ac.vel_ned
            lk = self.link(k, target, float(np.arctan2(vn[1], vn[0])), getattr(a, "rcs", 1.0))
            if not lk["in_beam"]:
                continue
            self._sweep[k][1] += 1
            if self.rng.random() > lk["pd"]:
                continue
            sig = rd.sigmas(lk["snr"])
            rm = lk["r"] + rd.range_bias + self.rng.normal(0, sig[0])
            am = np.radians(lk["az"] + rd.az_bias_deg) + self.rng.normal(0, sig[1])
            em = np.radians(lk["el"]) + self.rng.normal(0, sig[2])
            dets.append(Detection(t, k, polar_to_ned(pos, rm, am, em), measurement_cov(rm, am, em, sig),
                                  a.name))
        return dets

    def _false_alarms(self, t, k, rd, u0, u, wrap) -> list[Detection]:
        du = (u - u0) % 1.0
        lam = rd.false_alarms * du                           # falsos alarmes por varredura, proporcional ao feixe
        out = []
        pos = (rd.n, rd.e, self._ant[k])
        sig = rd.sigmas(SNR_REF)
        for _ in range(self.rng.poisson(lam)):
            r = float(max(rd.min_range, rd.range_m * self.rng.random()))      # clutter: denso perto do radar
            w = min(rd.az_width_deg, 360.0)
            az = np.radians(rd.az_start + w * ((u0 + self.rng.random() * du) % 1.0))
            el = np.radians(self.rng.uniform(max(rd.min_elev_deg, 0.0), min(rd.max_elev_deg, 15.0)))
            out.append(Detection(t, k, polar_to_ned(pos, r, az, el), measurement_cov(r, az, el, sig), None))
        return out

    def _log_track(self, t: float, tr: Track, agents):
        best, d_best = None, np.inf
        for a in agents:
            if not a.alive:
                continue
            d = float(np.linalg.norm(tr.x[:3] - np.array([a.aircraft.x[0], a.aircraft.x[1], a.aircraft.alt])))
            if d < d_best:
                best, d_best = a.name, d
        truth = best if d_best < 1000.0 else None
        self.track_log.setdefault(tr.id, []).append(TrackSample(
            t, tr.x.copy(), float(np.sqrt(np.trace(tr.P[:3, :3]) / 3.0)), tr.confirmed, truth,
            d_best if truth else float("nan"), tr.hits, tr.t_born))

    # ------------------------------------------------------------- consulta
    def picture(self, t: float, max_age: float | None = None) -> list[dict]:
        """Quadro tatico em t: pistas confirmadas (extrapoladas a velocidade constante ate t).

        Esta e a unica informacao de posicao das aeronaves que um controlador deve usar."""
        max_age = self.tracker.drop_after if max_age is None else max_age
        out = []
        for tid, log in self.track_log.items():
            ts = [s.t for s in log]
            i = bisect_right(ts, t) - 1
            if i < 0 or not log[i].confirmed or t - log[i].t > max_age:
                continue
            s = log[i]
            x = s.x.copy()
            x[:3] += x[3:] * (t - s.t)
            out.append({"id": tid, "n": x[0], "e": x[1], "alt": x[2], "vn": x[3], "ve": x[4], "vh": x[5],
                        "age": t - s.t, "sigma": s.sigma, "hits": s.hits, "life": t - s.born, "truth": s.truth})
        return out

    def detections_between(self, t0: float, t1: float) -> list[Detection]:
        return [d for d in self.detections if t0 < d.t <= t1]

    # ------------------------------------------------------------- metricas
    def summary(self, names: list[str], t_end: float, dt_grid: float = 1.0) -> list[dict]:
        rows = []
        grid = np.arange(0.0, t_end, dt_grid)
        for name in names:
            samples = sorted((s for log in self.track_log.values() for s in log
                              if s.confirmed and s.truth == name), key=lambda s: s.t)
            n_det = [sum(1 for d in self.detections if d.truth == name and d.radar == k)
                     for k in range(len(self.radars))]
            if not samples:
                rows.append({"agent": name, "rastreada": 0.0, "t_1a_pista": None, "rmse_pos": None,
                             "deteccoes": n_det})
                continue
            ts = np.array([s.t for s in samples])
            idx = np.searchsorted(ts, grid, side="right") - 1
            age = np.where(idx >= 0, grid - ts[np.clip(idx, 0, None)], np.inf)
            err = np.array([s.err for s in samples])
            rows.append({"agent": name, "rastreada": float(np.mean(age <= self.tracker.drop_after)),
                         "t_1a_pista": float(ts[0]), "rmse_pos": float(np.sqrt(np.nanmean(err ** 2))),
                         "deteccoes": n_det})
        return rows

    def error_series(self, name: str):
        """(t, erro 3D [m]) das estimativas confirmadas associadas a ``name``."""
        s = sorted((x for log in self.track_log.values() for x in log if x.confirmed and x.truth == name),
                   key=lambda x: x.t)
        return np.array([x.t for x in s]), np.array([x.err for x in s])
