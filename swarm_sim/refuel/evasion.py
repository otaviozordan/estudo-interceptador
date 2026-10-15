"""Manobras evasivas das aeronaves e leis de guiagem dos drones (ambiente de reabastecimento).

Evasao: a aeronave segue a rota da missao com um DESLOCAMENTO LATERAL y (perpendicular a rota) de dinamica de 2a
ordem com aceleracao limitada (``a_max``), mais uma variacao de velocidade ao longo da rota. O perfil (``evasion``
no ``AircraftSpec``) combina:

* ``weave``  [A_min, A_max] m + ``period`` [s]: zigue-zague senoidal;
* ``jink``   [A_min, A_max] m + ``jink_dt`` [s]: alvo lateral aleatorio trocado em intervalos aleatorios;
* ``react``  R m: alerta de aproximacao; se um drone esta a menos de R m e o tempo ate o encontro e menor que
  ``react_tgo`` s, quebra com aceleracao maxima para o lado oposto ao drone durante ``react_hold`` s;
* ``speed_var`` f: velocidade sorteada em [1-f, 1+f] x nominal, trocada a cada 8..20 s (rampa de 1 m/s2).

O deslocamento some perto do impacto (|y| <= 0.3 x distancia restante), entao o ponto de queda nao muda.
Os parametros sorteados (amplitude, periodo, fase) vem do gerador do episodio.

Guiagem (drones, plano horizontal, velocidade constante, aceleracao lateral <= ``amax``):
``pp`` perseguicao pura, ``pn`` navegacao proporcional verdadeira, ``apn`` PN aumentada (com a aceleracao do alvo
medida pelo sensor proprio). No transito o drone persegue o ponto previsto de interceptacao da pista de radar.
"""
from __future__ import annotations

import numpy as np

EV_DT = 0.25                    # passo interno da dinamica lateral [s]
EV_WN, EV_ZETA = 0.7, 0.8       # banda e amortecimento do rastreamento do deslocamento lateral


def _rng_range(rng, v, default=None):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    return float(rng.uniform(v[0], v[1]))


class Evasion:
    """Estado da manobra evasiva de UMA aeronave num episodio."""

    def __init__(self, spec: dict, rng: np.random.Generator):
        self.rng = rng
        self.a_max = _rng_range(rng, spec.get("a_max"), 10.0)
        self.y_max = _rng_range(rng, spec.get("y_max"), 400.0)
        self.weave = _rng_range(rng, spec.get("weave"), 0.0)
        self.period = _rng_range(rng, spec.get("period"), 15.0)
        self.phase = float(rng.uniform(0, 2 * np.pi))
        self.jink = _rng_range(rng, spec.get("jink"), 0.0)
        self.jink_dt = spec.get("jink_dt", [3.0, 8.0])
        self.react = _rng_range(rng, spec.get("react"), 0.0)
        self.react_tgo = float(spec.get("react_tgo", 4.0))
        self.react_hold = float(spec.get("react_hold", 3.0))
        self.speed_var = _rng_range(rng, spec.get("speed_var"), 0.0)
        self.y = self.yd = 0.0
        self.jref, self.jnext = 0.0, 0.0
        self.mult, self.mref, self.mnext = 1.0, 1.0, 0.0
        self.brk_sign, self.brk_until = 0.0, -1.0
        self.breaks = 0

    def reference(self, t: float) -> float:
        y = self.weave * np.sin(2 * np.pi * t / self.period + self.phase) if self.weave > 0 else 0.0
        if self.jink > 0:
            if t >= self.jnext:
                self.jref = float(self.rng.uniform(-self.jink, self.jink))
                self.jnext = t + float(self.rng.uniform(*self.jink_dt))
            y += self.jref
        return y

    def alert(self, t: float, rel_threat: np.ndarray | None, closing: float, normal: np.ndarray):
        """rel_threat: posicao do drone mais ameacador relativa a aeronave (ou None); closing: velocidade de
        aproximacao [m/s]. Dispara a quebra se o encontro esta proximo."""
        if self.react <= 0 or rel_threat is None or t < self.brk_until:
            return
        d = float(np.linalg.norm(rel_threat))
        if d <= self.react and closing > 1.0 and d / closing <= self.react_tgo:
            side = float(rel_threat @ normal)
            self.brk_sign = -1.0 if side > 0 else 1.0           # para o lado oposto ao drone
            self.brk_until = t + self.react_hold
            self.breaks += 1

    def step(self, t: float, h: float, y_lim: float, v_along: float) -> tuple[float, float, float]:
        """Integra ate t+h. Devolve (y, y', multiplicador de velocidade)."""
        n = max(1, int(np.ceil(h / EV_DT - 1e-9)))
        dt = h / n
        lim = min(self.y_max, y_lim)
        for i in range(n):
            ti = t + i * dt
            if ti < self.brk_until:
                u = self.brk_sign * self.a_max
            else:
                yr = float(np.clip(self.reference(ti), -lim, lim))
                u = EV_WN ** 2 * (yr - self.y) - 2 * EV_ZETA * EV_WN * self.yd
                u = float(np.clip(u, -self.a_max, self.a_max))
            self.yd = float(np.clip(self.yd + u * dt, -0.8 * v_along, 0.8 * v_along))
            self.y += self.yd * dt
            if abs(self.y) > lim:                                 # fronteira (inclui o afunilamento no impacto)
                self.y = float(np.sign(self.y) * lim)
                self.yd = 0.0 if self.y * self.yd > 0 else self.yd
            if self.speed_var > 0:
                if ti >= self.mnext:
                    self.mref = float(self.rng.uniform(1 - self.speed_var, 1 + self.speed_var))
                    self.mnext = ti + float(self.rng.uniform(8.0, 20.0))
                dm = 1.0 / max(v_along, 1.0) * dt
                self.mult += float(np.clip(self.mref - self.mult, -dm, dm))
        return self.y, self.yd, self.mult


def make_evasion(spec: dict | None, rng: np.random.Generator) -> Evasion | None:
    if not spec:
        return None
    return Evasion(spec, rng)


# ================================================================================ guiagem dos drones
def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def turn_toward(vel: np.ndarray, desired_dir: np.ndarray, amax: float, h: float) -> np.ndarray:
    """Gira o vetor velocidade (modulo constante) em direcao a desired_dir com taxa <= amax/|v|."""
    sp = float(np.linalg.norm(vel))
    if sp < 1e-9 or np.linalg.norm(desired_dir) < 1e-9:
        return vel
    psi = np.arctan2(vel[1], vel[0])
    psd = np.arctan2(desired_dir[1], desired_dir[0])
    dpsi = (psd - psi + np.pi) % (2 * np.pi) - np.pi
    w = amax / sp * h
    psi += float(np.clip(dpsi, -w, w))
    return sp * np.array([np.cos(psi), np.sin(psi)])


def lateral_accel(vel: np.ndarray, a_perp: float, amax: float, h: float) -> np.ndarray:
    """Aplica aceleracao lateral (perpendicular a velocidade, positiva a esquerda) limitada a amax."""
    sp = float(np.linalg.norm(vel))
    if sp < 1e-9:
        return vel
    a = float(np.clip(a_perp, -amax, amax))
    psi = np.arctan2(vel[1], vel[0]) + a / sp * h
    return sp * np.array([np.cos(psi), np.sin(psi)])


MPC_N, MPC_DT, MPC_LAMBDA = 12, 0.25, 1e-3      # horizonte (passos), passo [s] e peso do esforco do MPC
_MPC_CACHE: dict = {}


def _mpc_matrices(N: int, dt: float):
    """Posicoes do integrador duplo ao longo do horizonte: p_k = p0 + v0 t_k + sum_m G[k, m] a_m (por eixo)."""
    key = (N, dt)
    if key not in _MPC_CACHE:
        G = np.zeros((N, N))
        for k in range(N):
            for m in range(k + 1):
                G[k, m] = (k - m + 0.5) * dt * dt
        _MPC_CACHE[key] = (G, dt * np.arange(1, N + 1))
    return _MPC_CACHE[key]


def mpc_accel(pos, vel, tpos, tvel, amax: float, N: int = MPC_N, dt: float = MPC_DT,
              lam: float = MPC_LAMBDA) -> np.ndarray:
    """Guiagem preditiva (no espirito de Cruz, 2025): preve o alvo com VELOCIDADE CONSTANTE e escolhe a sequencia de
    aceleracoes a_0..a_{N-1} (|a_x|,|a_y| <= amax) que minimiza sum_k ||p_k - r_k||^2 + lam ||a||^2 (QP com caixa,
    resolvido por minimos quadrados limitados). Devolve a_0 (horizonte deslizante)."""
    from scipy.optimize import lsq_linear
    G, t = _mpc_matrices(N, dt)
    pos, vel, tpos, tvel = (np.asarray(x, float) for x in (pos, vel, tpos, tvel))
    rows = []
    rhs = []
    Z = np.zeros((N, N))
    for ax in range(2):                                        # variaveis: [a_x(0..N-1), a_y(0..N-1)]
        free = pos[ax] + vel[ax] * t
        ref = tpos[ax] + tvel[ax] * t
        rows.append(np.hstack([G, Z]) if ax == 0 else np.hstack([Z, G]))
        rhs.append(ref - free)
    A = np.vstack(rows + [np.sqrt(lam) * np.eye(2 * N)])
    b = np.concatenate(rhs + [np.zeros(2 * N)])
    sol = lsq_linear(A, b, bounds=(-amax, amax), method="bvls")
    return np.array([sol.x[0], sol.x[N]])


def guidance_step(law: str, pos, vel, tpos, tvel, tacc, amax: float, h: float, N: float = 4.0) -> np.ndarray:
    """Nova velocidade do drone na fase terminal (alvo com posicao/velocidade/aceleracao do sensor proprio)."""
    r = np.asarray(tpos) - pos
    if law == "pp":
        return turn_toward(vel, r, amax, h)
    if law == "mpc":                                           # drone de velocidade constante: usa a componente
        a = mpc_accel(pos, vel, tpos, tvel, amax)              # lateral da 1a aceleracao do plano
        u = vel / max(np.linalg.norm(vel), 1e-9)
        return lateral_accel(vel, float(a @ _perp(u)), amax, h)
    vr = np.asarray(tvel) - vel
    r2 = float(r @ r)
    if r2 < 1e-6:
        return vel
    lam_dot = float(r[0] * vr[1] - r[1] * vr[0]) / r2          # taxa da linha de visada (z, positiva a esquerda)
    vc = -float(r @ vr) / np.sqrt(r2)                          # velocidade de aproximacao
    if vc <= 0:                                                # afastando (passou do alvo): realinha por perseguicao
        return turn_toward(vel, r, amax, h)
    a = N * vc * lam_dot                                       # comando perpendicular a linha de visada
    if law == "apn" and tacc is not None:
        n_los = _perp(r / np.sqrt(r2))
        a += 0.5 * N * float(np.asarray(tacc) @ n_los)
    # converte o comando (normal a LOS) para normal a velocidade do drone
    u = vel / max(np.linalg.norm(vel), 1e-9)
    cos_l = float(u @ (r / np.sqrt(r2)))
    a_perp = a / max(cos_l, 0.3)
    return lateral_accel(vel, a_perp, amax, h)


def closest_approach(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Menor |r0 + (r1 - r0) s|, s em [0,1] (movimento relativo linear no subpasso). r0, r1: (..., 2)."""
    d = r1 - r0
    dd = np.sum(d * d, -1)
    s = np.clip(np.where(dd > 1e-12, -np.sum(r0 * d, -1) / np.maximum(dd, 1e-12), 0.0), 0.0, 1.0)
    return np.linalg.norm(r0 + d * s[..., None], axis=-1)
