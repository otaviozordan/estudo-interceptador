"""Piloto automatico em cascata (rolagem/arfagem/velocidade) para o UAV de asa fixa.

Entrada: aceleracao lateral (do guiado L1), altitude MSL e velocidade comandadas.
  lateral  : a_lat -> phi_cmd = atan(a_lat/g) -> aileron (PD) ; leme = amortecedor de guinada
  vertical : h_cmd -> razao de subida -> gamma_cmd -> theta_cmd (PI) -> profundor (PD)
  potencia : velocidade do ar -> manete (PI + feed-forward de subida)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import G, Controls, FixedWing, Trim, isa_density
from .guidance import GuidanceCmd


@dataclass
class AutopilotGains:
    kp_phi: float = 0.9
    kd_phi: float = 0.2
    kr_yaw: float = 0.5
    kp_theta: float = 1.2
    kd_theta: float = 0.25
    kp_h: float = 0.12            # altitude -> razao de subida [1/s]
    kp_gamma: float = 1.2
    ki_gamma: float = 0.35
    kp_v: float = 0.06
    ki_v: float = 0.02
    phi_max: float = np.radians(35.0)
    theta_max: float = np.radians(25.0)
    climb_max: float = 7.5        # [m/s]
    sink_max: float = 7.0         # [m/s]
    v_prot_margin: float = 3.0    # abaixo disso limita a subida (protecao de velocidade)


class Autopilot:
    def __init__(self, aircraft: FixedWing, trim: Trim, dt: float,
                 gains: AutopilotGains | None = None):
        self.ac, self.trim, self.dt = aircraft, trim, dt
        self.g = gains or AutopilotGains()
        self.int_gamma = 0.0
        self.int_v = 0.0
        self.r_lp = 0.0
        self.hdot_cmd = 0.0
        self.phi_cmd = 0.0
        self.v_ref = trim.Va                      # comando de velocidade apos a rampa
        self.v_rate = 2.0                         # [m/s^2] taxa maxima de variacao do comando
        grid = np.arange(18.0, 40.1, 2.0)
        tr_tab = [aircraft.trim(v, 0.0, max(aircraft.alt, 0.0)) for v in grid]
        self._tab = (grid, np.array([t.alpha for t in tr_tab]), np.array([t.de for t in tr_tab]),
                     np.array([t.dt for t in tr_tab]))

    def trim_at(self, v: float) -> tuple[float, float, float]:
        """(alpha, de, dt) de equilibrio interpolados para a velocidade v."""
        g, al, de, dt = self._tab
        return (float(np.interp(v, g, al)), float(np.interp(v, g, de)), float(np.interp(v, g, dt)))

    def update(self, cmd: GuidanceCmd) -> Controls:
        g, ac, tr, dt = self.g, self.ac, self.trim, self.dt
        phi, theta, psi = ac.euler
        p, q, r = ac.x[10:13]
        Va, alpha, beta = ac.air_data()
        vn = ac.vel_ned
        hdot = -vn[2]
        Vg = max(np.linalg.norm(vn), 1.0)
        gamma = np.arcsin(np.clip(hdot / Vg, -1, 1))
        alt = ac.alt

        # ---------------- lateral
        phi_c = np.clip(np.arctan2(cmd.a_lat, G), -g.phi_max, g.phi_max)
        da = g.kp_phi * (phi_c - phi) - g.kd_phi * p
        r_coord = G * np.sin(phi) * np.cos(theta) / max(Va, 5.0)     # guinada de curva coordenada
        dr = g.kr_yaw * (r - r_coord)

        # ---------------- vertical
        self.v_ref += float(np.clip(cmd.speed_cmd - self.v_ref, -self.v_rate * dt, self.v_rate * dt))
        a_tr, de_tr, dt_tr = self.trim_at(self.v_ref)
        v_err = self.v_ref - Va
        hdot_hi = g.climb_max
        if v_err > g.v_prot_margin:                                   # velocidade baixa: nao subir
            hdot_hi = max(0.0, g.climb_max * (1 - (v_err - g.v_prot_margin) / 5.0))
        hdot_c = float(np.clip(g.kp_h * (cmd.alt_cmd - alt), -g.sink_max, hdot_hi))
        gamma_c = np.arcsin(np.clip(hdot_c / max(Va, 10.0), -0.6, 0.6))
        e_g = gamma_c - gamma
        self.int_gamma = float(np.clip(self.int_gamma + g.ki_gamma * e_g * dt, -0.15, 0.15))
        theta_c = np.clip(a_tr + gamma_c + g.kp_gamma * e_g + self.int_gamma,
                          -g.theta_max, g.theta_max)
        de = de_tr - (g.kp_theta * (theta_c - theta) - g.kd_theta * q)

        # ---------------- potencia
        rho = isa_density(alt)
        dTddt = max(rho * ac.p.S_prop * ac.p.C_prop * ac.p.k_motor**2 * max(dt_tr, 0.2), 1.0)
        ff = ac.p.mass * G * np.sin(gamma_c) / dTddt
        dt_unsat = dt_tr + ff + g.kp_v * v_err + self.int_v
        dt_cmd = float(np.clip(dt_unsat, 0.0, 1.0))
        if dt_unsat == dt_cmd:                                        # anti-windup
            self.int_v = float(np.clip(self.int_v + g.ki_v * v_err * dt, -0.4, 0.4))

        self.hdot_cmd = hdot_c
        self.phi_cmd = float(phi_c)
        return Controls(de=float(de), da=float(da), dr=float(dr), dt=dt_cmd)
