"""Zonas de rajada de vento: a aeronave sente o vento adicional enquanto estiver dentro do raio da zona."""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class GustZone:
    n: float
    e: float
    radius: float = 400.0       # raio de influencia [m]
    speed: float = 10.0         # velocidade horizontal no centro [m/s]
    dir_deg: float = 90.0       # rumo PARA ONDE o vento sopra (0 = norte, 90 = leste)
    up: float = 0.0             # componente vertical no centro [m/s] (+ = ascendente)

    def weight(self, n: float, e: float) -> float:
        """1 no centro, 0 na borda (janela cosseno), 0 fora."""
        d = float(np.hypot(n - self.n, e - self.e))
        return 0.5 * (1.0 + np.cos(np.pi * d / self.radius)) if d < self.radius else 0.0

    def vector(self, n: float, e: float) -> np.ndarray:
        """Vento NED [m/s] na posicao (n, e)."""
        w, h = self.weight(n, e), np.radians(self.dir_deg)
        return w * np.array([self.speed * np.cos(h), self.speed * np.sin(h), -self.up])


def gust_wind(zones, n: float, e: float) -> np.ndarray:
    v = np.zeros(3)
    for z in zones:
        v += z.vector(n, e)
    return v


_FIELDS = {f.name for f in fields(GustZone)}


def gust_from_dict(d: dict) -> GustZone:
    return GustZone(**{k: v for k, v in d.items() if k in _FIELDS})
