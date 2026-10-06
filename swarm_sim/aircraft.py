"""UAV de asa fixa 6-DOF (corpo rigido, quaternions, NED).

Modelo aerodinamico baseado em coeficientes de estabilidade (Aerosonde,
Beard & McLain) com polar parabolica, estol suave (blending), empuxo de helice
e densidade ISA. Estado: [pn pe pd u v w e0 e1 e2 e3 p q r].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

G = 9.80665


def isa_density(alt_msl: float) -> float:
    h = min(max(alt_msl, -500.0), 11000.0)
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Matriz de rotacao corpo -> NED."""
    e0, e1, e2, e3 = q
    return np.array([
        [e1**2 + e0**2 - e2**2 - e3**2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
        [2 * (e1 * e2 + e3 * e0), e2**2 + e0**2 - e1**2 - e3**2, 2 * (e2 * e3 - e1 * e0)],
        [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e3**2 + e0**2 - e1**2 - e2**2],
    ])


def euler_to_quat(phi, theta, psi) -> np.ndarray:
    c0, c1, c2 = np.cos([phi / 2, theta / 2, psi / 2])
    s0, s1, s2 = np.sin([phi / 2, theta / 2, psi / 2])
    return np.array([
        c0 * c1 * c2 + s0 * s1 * s2,
        s0 * c1 * c2 - c0 * s1 * s2,
        c0 * s1 * c2 + s0 * c1 * s2,
        c0 * c1 * s2 - s0 * s1 * c2,
    ])


def quat_to_euler(q) -> tuple[float, float, float]:
    e0, e1, e2, e3 = q
    phi = np.arctan2(2 * (e0 * e1 + e2 * e3), e0**2 + e3**2 - e1**2 - e2**2)
    theta = np.arcsin(np.clip(2 * (e0 * e2 - e1 * e3), -1, 1))
    psi = np.arctan2(2 * (e0 * e3 + e1 * e2), e0**2 + e1**2 - e2**2 - e3**2)
    return phi, theta, psi


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class AircraftParams:
    mass: float = 13.5
    Jx: float = 0.8244
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.1204
    S: float = 0.55
    b: float = 2.8956
    c: float = 0.18994
    e_oswald: float = 0.9
    # sustentacao / arrasto / momento de arfagem
    CL0: float = 0.23
    CLa: float = 5.61
    CLq: float = 7.95
    CLde: float = 0.13
    CDp: float = 0.043
    CDde: float = 0.0135
    Cm0: float = 0.0135
    Cma: float = -2.74
    Cmq: float = -38.21
    Cmde: float = -0.99
    alpha_stall: float = 0.4712
    blend_M: float = 50.0
    # lateral-direcional
    CY0: float = 0.0
    CYb: float = -0.98
    CYp: float = 0.0
    CYr: float = 0.0
    CYda: float = 0.0
    CYdr: float = -0.17
    Cl0: float = 0.0
    Clb: float = -0.13
    Clp: float = -0.51
    Clr: float = 0.25
    Clda: float = 0.17
    Cldr: float = 0.0024
    Cn0: float = 0.0
    Cnb: float = 0.073
    Cnp: float = 0.069
    Cnr: float = -0.095
    Cnda: float = -0.011
    Cndr: float = -0.069
    # propulsao: T = 0.5 rho S_prop C_prop ((k_motor*dt)^2 - Va^2)
    S_prop: float = 0.2027
    C_prop: float = 1.0
    k_motor: float = 45.0
    # atuadores
    surf_limit: float = np.radians(30.0)
    tau_surf: float = 0.05
    tau_thr: float = 0.25

    @property
    def AR(self) -> float:
        return self.b**2 / self.S


@dataclass
class Controls:
    de: float = 0.0   # profundor [rad] (+ = bordo de fuga p/ baixo -> nariz p/ baixo)
    da: float = 0.0   # aileron [rad] (+ = rolar p/ a direita)
    dr: float = 0.0   # leme [rad]
    dt: float = 0.0   # manete [0..1]

    def as_array(self):
        return np.array([self.de, self.da, self.dr, self.dt])


@dataclass
class Trim:
    Va: float
    gamma: float
    alpha: float
    de: float
    dt: float


class Wind:
    """Vento medio NED + rajada Gauss-Markov de 1a ordem."""

    def __init__(self, steady=(0.0, 0.0, 0.0), gust_sigma=0.0, gust_L=200.0, seed=0):
        self.steady = np.asarray(steady, float)
        self.sigma, self.L = gust_sigma, gust_L
        self.gust = np.zeros(3)
        self.rng = np.random.default_rng(seed)

    def step(self, Va: float, dt: float):
        if self.sigma > 0:
            a = np.exp(-max(Va, 1.0) / self.L * dt)
            self.gust = a * self.gust + self.sigma * np.sqrt(1 - a * a) * self.rng.standard_normal(3)
        return self.steady + self.gust


class FixedWing:
    def __init__(self, params: AircraftParams | None = None):
        self.p = params or AircraftParams()
        P = self.p
        Gm = P.Jx * P.Jz - P.Jxz**2
        self._g = (P.Jxz * (P.Jx - P.Jy + P.Jz) / Gm, (P.Jz * (P.Jz - P.Jy) + P.Jxz**2) / Gm,
                   P.Jz / Gm, P.Jxz / Gm, (P.Jz - P.Jx) / P.Jy, P.Jxz / P.Jy,
                   ((P.Jx - P.Jy) * P.Jx + P.Jxz**2) / Gm, P.Jx / Gm)
        self.x = np.zeros(13)
        self.x[6] = 1.0
        self.u = Controls()       # comandos
        self.act = np.zeros(4)    # posicao real dos atuadores
        self.wind_ned = np.zeros(3)

    # ------------------------------------------------------------- inicializacao
    def reset(self, n, e, alt_msl, psi, Va, gamma=0.0, trim: Trim | None = None):
        trim = trim or self.trim(Va, gamma, alt_msl)
        theta = trim.alpha + gamma
        q = euler_to_quat(0.0, theta, psi)
        self.x = np.concatenate([[n, e, -alt_msl],
                                 [Va * np.cos(trim.alpha), 0.0, Va * np.sin(trim.alpha)],
                                 q, np.zeros(3)])
        self.act = np.array([trim.de, 0.0, 0.0, trim.dt])
        self.u = Controls(*self.act)
        self.wind_ned = np.zeros(3)
        return trim

    # --------------------------------------------------------------- acessores
    @property
    def pos(self): return self.x[0:3]
    @property
    def alt(self): return -self.x[2]
    @property
    def euler(self): return quat_to_euler(self.x[6:10])
    @property
    def vel_ned(self): return quat_to_rot(self.x[6:10]) @ self.x[3:6]

    def air_data(self, x=None, wind_ned=None):
        x = self.x if x is None else x
        w = self.wind_ned if wind_ned is None else wind_ned
        v_air = x[3:6] - quat_to_rot(x[6:10]).T @ w
        Va = np.linalg.norm(v_air)
        alpha = np.arctan2(v_air[2], v_air[0])
        beta = np.arcsin(np.clip(v_air[1] / max(Va, 1e-3), -1, 1))
        return Va, alpha, beta

    # --------------------------------------------------------------- dinamica
    def forces_moments(self, x, act, wind_ned):
        P = self.p
        de, da, dr, dt = act
        Va, alpha, beta = self.air_data(x, wind_ned)
        Va_s = max(Va, 1.0)
        rho = isa_density(-x[2])
        qbar = 0.5 * rho * Va_s**2
        p_, q_, r_ = x[10:13]
        # estol suave
        e1 = np.exp(np.clip(-P.blend_M * (alpha - P.alpha_stall), -50, 50))
        e2 = np.exp(np.clip(P.blend_M * (alpha + P.alpha_stall), -50, 50))
        sigma = (1 + e1 + e2) / ((1 + e1) * (1 + e2))
        CL_lin = P.CL0 + P.CLa * alpha
        CL_stall = 2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
        CL = (1 - sigma) * CL_lin + sigma * CL_stall + P.CLq * P.c / (2 * Va_s) * q_ + P.CLde * de
        CD = (P.CDp + CL**2 / (np.pi * P.e_oswald * P.AR) + P.CDde * abs(de)
              + sigma * 2 * np.sin(alpha) ** 2)
        L, D = qbar * P.S * CL, qbar * P.S * CD
        ca, sa = np.cos(alpha), np.sin(alpha)
        bb = P.b / (2 * Va_s)
        f_aero = np.array([-D * ca + L * sa,
                           qbar * P.S * (P.CY0 + P.CYb * beta + P.CYdr * dr + P.CYda * da
                                         + (P.CYp * p_ + P.CYr * r_) * bb),
                           -D * sa - L * ca])
        T = 0.5 * rho * P.S_prop * P.C_prop * ((P.k_motor * dt) ** 2 - Va**2)
        grav = quat_to_rot(x[6:10]).T @ np.array([0, 0, P.mass * G])
        F = f_aero + np.array([T, 0, 0]) + grav
        l = qbar * P.S * P.b * (P.Cl0 + P.Clb * beta + P.Clp * bb * p_ + P.Clr * bb * r_
                                + P.Clda * da + P.Cldr * dr)
        m = qbar * P.S * P.c * (P.Cm0 + P.Cma * alpha + P.Cmq * P.c / (2 * Va_s) * q_ + P.Cmde * de)
        n = qbar * P.S * P.b * (P.Cn0 + P.Cnb * beta + P.Cnp * bb * p_ + P.Cnr * bb * r_
                                + P.Cnda * da + P.Cndr * dr)
        return F, np.array([l, m, n]), T

    def deriv(self, x, act, wind_ned):
        P = self.p
        g1, g2, g3, g4, g5, g6, g7, g8 = self._g
        F, M, _ = self.forces_moments(x, act, wind_ned)
        u, v, w = x[3:6]
        p, q, r = x[10:13]
        e0, e1, e2, e3 = x[6:10]
        d = np.empty(13)
        d[0:3] = quat_to_rot(x[6:10]) @ x[3:6]
        d[3] = r * v - q * w + F[0] / P.mass
        d[4] = p * w - r * u + F[1] / P.mass
        d[5] = q * u - p * v + F[2] / P.mass
        d[6] = 0.5 * (-p * e1 - q * e2 - r * e3)
        d[7] = 0.5 * (p * e0 + r * e2 - q * e3)
        d[8] = 0.5 * (q * e0 - r * e1 + p * e3)
        d[9] = 0.5 * (r * e0 + q * e1 - p * e2)
        d[10] = g1 * p * q - g2 * q * r + g3 * M[0] + g4 * M[2]
        d[11] = g5 * p * r - g6 * (p**2 - r**2) + M[1] / P.Jy
        d[12] = g7 * p * q - g1 * q * r + g4 * M[0] + g8 * M[2]
        return d

    def step(self, dt: float, wind_ned=None):
        """Avanca dt [s] com RK4; atuadores de 1a ordem com saturacao."""
        P = self.p
        if wind_ned is not None:
            self.wind_ned = np.asarray(wind_ned, float)
        lim = P.surf_limit
        c = self.u.as_array()
        cmd = np.array([np.clip(c[0], -lim, lim), np.clip(c[1], -lim, lim),
                        np.clip(c[2], -lim, lim), np.clip(c[3], 0.0, 1.0)])
        tau = np.array([P.tau_surf] * 3 + [P.tau_thr])
        self.act = cmd + (self.act - cmd) * np.exp(-dt / tau)

        x, a, w = self.x, self.act, self.wind_ned
        k1 = self.deriv(x, a, w)
        k2 = self.deriv(x + 0.5 * dt * k1, a, w)
        k3 = self.deriv(x + 0.5 * dt * k2, a, w)
        k4 = self.deriv(x + dt * k3, a, w)
        self.x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

    # ------------------------------------------------------------------- trim
    def trim(self, Va: float, gamma: float = 0.0, alt_msl: float = 1000.0) -> Trim:
        """Voo reto (nivelado ou subida/descida constante): resolve (alpha, de, dt)."""
        def resid(z):
            alpha, de, dt = z
            x = np.concatenate([[0, 0, -alt_msl],
                                [Va * np.cos(alpha), 0, Va * np.sin(alpha)],
                                euler_to_quat(0, alpha + gamma, 0), np.zeros(3)])
            d = self.deriv(x, np.array([de, 0, 0, dt]), np.zeros(3))
            return [d[3], d[5], d[11]]

        sol = fsolve(resid, [0.05, -0.05, 0.5], xtol=1e-12)
        return Trim(Va, gamma, *sol)
