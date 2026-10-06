"""Imagem de satelite (tiles XYZ, Esri World Imagery) alinhada ao DEM local.

Os tiles sao baixados sob demanda e guardados em data/tiles/ (nao versionado).
Imagery (c) Esri, Maxar, Earthstar Geographics - uso para pesquisa/planejamento; respeite os termos do provedor.
"""
from __future__ import annotations

import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .terrain import M_PER_DEG_LAT, M_PER_DEG_LON_EQ, Terrain

URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
UA = "swarm-intercept-research/0.1"


def _merc(lat, lon, z):
    """(x, y) fracionarios de tile para lat/lon (arrays ok)."""
    n = 2.0**z
    lat = np.asarray(lat, float)
    return (np.asarray(lon, float) + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def _tile_range(terrain: Terrain, z: int):
    n_c = np.array([terrain.n0, terrain.n_max])
    e_c = np.array([terrain.e0, terrain.e_max])
    lat, lon = terrain.to_latlon(n_c, e_c)
    x, y = _merc(lat, lon, z)
    return int(np.floor(x.min())), int(np.floor(x.max())), int(np.floor(y.min())), int(np.floor(y.max()))


def auto_zoom(terrain: Terrain, max_tiles: int = 300, zmax: int = 17) -> int:
    for z in range(zmax, 7, -1):
        x0, x1, y0, y1 = _tile_range(terrain, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles:
            return z
    return 8


def _get_tile(z, x, y, cache_dir):
    path = os.path.join(cache_dir, str(z), str(x), f"{y}.jpg")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    for _ in range(3):
        try:
            req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return Image.open(path).convert("RGB")
        except Exception:
            continue
    return None


def fetch_satellite(terrain: Terrain, cache_dir: str, zoom: int | None = None, progress=None) -> Image.Image:
    """Mosaico de satelite cobrindo exatamente a extensao do terreno (linha 0 = norte)."""
    if terrain.lat0 is None:
        raise ValueError("terreno sem georreferencia (lat0/lon0): satelite indisponivel")
    z = zoom or auto_zoom(terrain)
    tx0, tx1, ty0, ty1 = _tile_range(terrain, z)
    coords = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (90, 90, 90))
    done = failed = 0
    with ThreadPoolExecutor(8) as ex:
        for (x, y), img in zip(coords, ex.map(lambda c: _get_tile(z, c[0], c[1], cache_dir), coords)):
            if img is None:
                failed += 1
            else:
                mosaic.paste(img, ((x - tx0) * 256, (y - ty0) * 256))
            done += 1
            if progress:
                progress(done, len(coords))
    if failed == len(coords):
        raise RuntimeError("nenhum tile baixado (sem internet?)")

    mpp = 156543.03392 * math.cos(math.radians(terrain.lat0)) / 2**z
    W = int(min((terrain.e_max - terrain.e0) / mpp, 6000))
    H = int(min((terrain.n_max - terrain.n0) / mpp, 6000))
    m_lon = M_PER_DEG_LON_EQ * math.cos(math.radians(terrain.lat0))
    e = terrain.e0 + (np.arange(W) + 0.5) * (terrain.e_max - terrain.e0) / W
    n = terrain.n_max - (np.arange(H) + 0.5) * (terrain.n_max - terrain.n0) / H
    lon = terrain.lon0 + e / m_lon
    lat = terrain.lat0 + n / M_PER_DEG_LAT
    xi = np.clip(((_merc(0, lon, z)[0] - tx0) * 256).astype(int), 0, mosaic.width - 1)
    yi = np.clip(((_merc(lat, 0, z)[1] - ty0) * 256).astype(int), 0, mosaic.height - 1)
    arr = np.asarray(mosaic)[yi[:, None], xi[None, :]]
    return Image.fromarray(np.ascontiguousarray(arr))
