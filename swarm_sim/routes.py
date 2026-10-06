"""Geradores de rotas (referencial local NED: n = norte, e = leste) [m]."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Waypoint:
    n: float
    e: float
    agl: float = 120.0                 # altura acima do solo [m]; <= 0 = waypoint no chao (impacto)
    speed: float | None = None         # velocidade do ar desejada [m/s] (None = cruzeiro)
    loiter_radius: float = 0.0         # >0: orbita em torno do ponto antes de seguir
    loiter_turns: float = 0.0          # numero de voltas na orbita
    loiter_dir: int = 1                # +1 horario, -1 anti-horario (visto de cima)

    @property
    def impact(self) -> bool:
        """Waypoint no chao (agl <= 0): o aviao mergulha ate ele e colide com o terreno."""
        return self.agl <= 0.0

    @classmethod
    def ground(cls, n: float, e: float, agl: float = 0.0, speed: float | None = None) -> "Waypoint":
        return cls(n, e, agl=agl, speed=speed)

    @property
    def ne(self) -> np.ndarray:
        return np.array([self.n, self.e])


@dataclass
class Route:
    name: str
    waypoints: list[Waypoint]
    loop: bool = False
    kind: str = "custom"

    def length(self) -> float:
        p = np.array([w.ne for w in self.waypoints])
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _cr_segment(p0, p1, p2, p3, m: int) -> np.ndarray:
    """Catmull-Rom centripeta (Barry-Goldman) de p1 a p2: m pontos, o ultimo e p2."""
    t = [0.0]
    for a, b in ((p0, p1), (p1, p2), (p2, p3)):
        t.append(t[-1] + max(float(np.linalg.norm(b - a)), 1e-6) ** 0.5)
    u = np.linspace(t[1], t[2], m + 1)[1:, None]
    lerp = lambda a, b, ta, tb: (tb - u) / (tb - ta) * a + (u - ta) / (tb - ta) * b
    a1, a2, a3 = lerp(p0, p1, t[0], t[1]), lerp(p1, p2, t[1], t[2]), lerp(p2, p3, t[2], t[3])
    b1, b2 = lerp(a1, a2, t[0], t[2]), lerp(a2, a3, t[1], t[3])
    return lerp(b1, b2, t[1], t[2])


def expand_spline(wps: list[Waypoint], start_ne=None, loop: bool = False,
                  spacing: float = 150.0) -> list[Waypoint]:
    """Suaviza a rota por uma spline que passa por todos os waypoints.

    Insere waypoints intermediarios a cada ~`spacing` m (AGL interpolado, velocidade do trecho de destino);
    os waypoints originais (inclusive loiter/impacto) sao mantidos. Em loop, a curva fecha do ultimo ao primeiro.
    `start_ne` entra como ancora da primeira curva (nao e incluido na saida).
    """
    closed = loop and len(wps) > 2
    nodes = list(wps)
    P = np.array([w.ne for w in nodes])
    if not closed and start_ne is not None:
        P = np.vstack([np.asarray(start_ne, float), P])
        nodes = [Waypoint(*start_ne, agl=wps[0].agl if wps[0].agl > 0 else 100.0)] + nodes
    n = len(P)
    if n < 2 or (n == 2 and not closed and start_ne is None):
        return list(wps)
    out: list[Waypoint] = []
    for i in range(n if closed else n - 1):
        a, b = nodes[i], nodes[(i + 1) % n]
        p1, p2 = P[i], P[(i + 1) % n]
        p0 = P[(i - 1) % n] if (closed or i > 0) else 2 * p1 - p2
        p3 = P[(i + 2) % n] if (closed or i + 2 < n) else 2 * p2 - p1
        m = max(int(np.ceil(np.linalg.norm(p2 - p1) / spacing)), 1)
        pts = _cr_segment(p0, p1, p2, p3, m)
        for k, q in enumerate(pts, start=1):
            if k == m:
                if not (closed and i == n - 1):          # fechamento: o primeiro waypoint ja existe
                    out.append(b)
            else:
                f = k / m
                out.append(Waypoint(float(q[0]), float(q[1]), agl=max(a.agl, 1.0) * (1 - f) + max(b.agl, 1.0) * f,
                                    speed=b.speed))
        if b.impact:
            break
    return out


def _rot(heading: float):
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]])   # (along, across) -> (n, e) para rumo `heading`


def patrol_route(center, length=6000.0, width=2500.0, heading_deg=0.0, agl=120.0,
                 speed=None, name="patrulha") -> Route:
    """Hipodromo (racetrack) fechado em loop em torno de `center`."""
    R = _rot(np.radians(heading_deg))
    corners = [(-length / 2, -width / 2), (length / 2, -width / 2),
               (length / 2, width / 2), (-length / 2, width / 2)]
    wps = [Waypoint(*(np.asarray(center) + R @ np.array(c)), agl=agl, speed=speed) for c in corners]
    return Route(name, wps, loop=True, kind="patrol")


def coverage_route(center, width=6000.0, height=4000.0, heading_deg=0.0, swath=500.0,
                   agl=120.0, speed=None, overrun=300.0, name="cobertura") -> Route:
    """Varredura 'boustrophedon' (lawn-mower) de um retangulo.

    As linhas de varredura seguem `heading_deg`; `swath` e o espacamento entre
    linhas e `overrun` estende cada linha alem da borda para o aviao fazer a curva.
    """
    R = _rot(np.radians(heading_deg))
    n_lines = max(2, int(np.ceil(height / swath)) + 1)
    across = np.linspace(-height / 2, height / 2, n_lines)
    wps = []
    for k, y in enumerate(across):
        x0, x1 = -width / 2 - overrun, width / 2 + overrun
        pts = [(x0, y), (x1, y)] if k % 2 == 0 else [(x1, y), (x0, y)]
        for x, yy in pts:
            wps.append(Waypoint(*(np.asarray(center) + R @ np.array([x, yy])), agl=agl, speed=speed))
    return Route(name, wps, loop=False, kind="coverage")


def inspection_route(points, offset=150.0, agl=80.0, speed=None, pois=(),
                     return_leg=True, name="inspecao") -> Route:
    """Inspecao de uma estrutura linear (linha de transmissao, duto, cume).

    Voa paralelamente a polilinha `points` deslocado `offset` m a direita, e
    (se `return_leg`) volta pelo outro lado. `pois` = [(n, e, raio, voltas)]
    insere orbitas de inspecao detalhada no ponto de rota mais proximo.
    """
    P = np.asarray(points, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    right = np.stack([-t[:, 1], t[:, 0]], axis=1)       # normal a direita (n, e)
    fwd = P + offset * right
    wps = [Waypoint(*p, agl=agl, speed=speed) for p in fwd]
    for (pn, pe, rad, turns) in pois:
        k = int(np.argmin(np.linalg.norm(fwd - [pn, pe], axis=1)))
        wps[k] = Waypoint(pn, pe, agl=agl, speed=speed, loiter_radius=rad, loiter_turns=turns)
    if return_leg:
        back = P[::-1] - offset * right[::-1]
        wps += [Waypoint(*p, agl=agl, speed=speed) for p in back]
    return Route(name, wps, loop=False, kind="inspection")
