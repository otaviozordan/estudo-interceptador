"""Marcos conhecidos (rotulos no mapa do construtor e do replay). Coordenadas WGS84 em graus decimais.

Fontes: Wikipedia/ANAC/sites institucionais; 'aprox.' = posicao aproximada (confira antes de usar como referencia).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    name: str
    lat: float
    lon: float
    kind: str = "ref"          # aeroporto | ensino | pesquisa | centro
    approx: bool = False


LANDMARKS = [
    # Sao Jose dos Campos
    Landmark("INPE", -23.2069, -45.8603, "pesquisa"),
    Landmark("ITA / DCTA", -23.2122, -45.8753, "ensino", approx=True),
    Landmark("Aeroporto SJK (SBSJ)", -23.2282, -45.8627, "aeroporto"),
    Landmark("Centro de S. J. dos Campos", -23.1800, -45.8900, "centro", approx=True),
    # Itajuba
    Landmark("UNIFEI", -22.4128, -45.4494, "ensino"),
    Landmark("Aeroporto de Itajubá (SBIT)", -22.4089, -45.4261, "aeroporto"),
    Landmark("Helibras (aeródromo)", -22.4308, -45.4792, "aeroporto"),        # portaria ANAC 0376/2015
    Landmark("Centro de Itajubá", -22.4226, -45.4529, "centro", approx=True),
    Landmark("Observatório Pico dos Dias (LNA)", -22.5344, -45.5825, "pesquisa"),
    # Pouso Alegre
    Landmark("Aeroporto de Pouso Alegre (SNZA)", -22.2889, -45.9194, "aeroporto"),
    Landmark("Centro de Pouso Alegre", -22.2300, -45.9360, "centro", approx=True),
    # Santa Rita do Sapucai
    Landmark("Inatel", -22.2572, -45.6961, "ensino"),
    Landmark("Centro de Santa Rita do Sapucaí", -22.2519, -45.7028, "centro", approx=True),
]

KIND_COLOR = {"aeroporto": "#9ad0ff", "ensino": "#ffe08a", "pesquisa": "#c3a6ff", "centro": "#e6e6e6", "ref": "#ffffff"}


def draw_landmarks(c, items, font=("Segoe UI", 8, "bold")):
    """Desenha os marcos (losango + rotulo com contorno escuro) num MapView. `items` = visible_landmarks(...)."""
    for lm, n, e in items:
        x, y = c.w2s(n, e)
        col = KIND_COLOR.get(lm.kind, "#ffffff")
        c.create_polygon(x, y - 6, x + 6, y, x, y + 6, x - 6, y, fill=col, outline="black", width=1)
        txt = lm.name + (" (aprox.)" if lm.approx else "")
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):            # contorno para ler sobre o satelite
            c.create_text(x + 9 + dx, y + dy, text=txt, fill="black", anchor="w", font=font)
        c.create_text(x + 9, y, text=txt, fill=col, anchor="w", font=font)


def visible_landmarks(terrain) -> list[tuple[Landmark, float, float]]:
    """Marcos dentro do mapa: [(marco, n, e)]. Vazio se o DEM e sintetico."""
    out = []
    for lm in LANDMARKS:
        ne = terrain.from_latlon(lm.lat, lm.lon)
        if ne is not None and bool(terrain.inside(*ne)):
            out.append((lm, float(ne[0]), float(ne[1])))
    return out
