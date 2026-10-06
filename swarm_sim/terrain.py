"""Terreno (DEM) em um referencial local NED.

O DEM e reamostrado para uma grade metrica uniforme centrada em (lat0, lon0):
``n`` cresce para o norte, ``e`` para o leste e a elevacao e MSL em metros.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


class Terrain:
    """Grade regular de elevacao com interpolacao bilinear.

    ``z[i, j]`` e a elevacao em ``n = n0 + i*res`` e ``e = e0 + j*res``.
    Fora da grade a altura e a da borda mais proxima (``inside`` indica o caso).
    """

    def __init__(self, z: np.ndarray, res: float, n0: float, e0: float,
                 lat0: float | None = None, lon0: float | None = None):
        self.z = np.asarray(z, dtype=np.float64)
        self.res = float(res)
        self.n0, self.e0 = float(n0), float(e0)
        self.lat0, self.lon0 = lat0, lon0
        self.ny, self.nx = self.z.shape
        self.n_max = self.n0 + (self.ny - 1) * self.res
        self.e_max = self.e0 + (self.nx - 1) * self.res

    # ------------------------------------------------------------------ consulta
    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(n_min, n_max, e_min, e_max)."""
        return self.n0, self.n_max, self.e0, self.e_max

    def inside(self, n, e):
        n, e = np.asarray(n), np.asarray(e)
        return (n >= self.n0) & (n <= self.n_max) & (e >= self.e0) & (e <= self.e_max)

    def height(self, n, e):
        """Elevacao MSL [m] (vetorizada, bilinear, clamp nas bordas)."""
        n, e = np.broadcast_arrays(np.asarray(n, float), np.asarray(e, float))
        i = np.clip((n - self.n0) / self.res, 0, self.ny - 1)
        j = np.clip((e - self.e0) / self.res, 0, self.nx - 1)
        h = map_coordinates(self.z, [i.ravel(), j.ravel()], order=1, mode="nearest")
        return h.reshape(n.shape) if n.ndim else float(h[0])

    def agl(self, n, e, alt_msl):
        """Altura acima do solo [m]."""
        return alt_msl - self.height(n, e)

    def slope(self, n, e):
        """Declividade do terreno [rad] (diferencas centrais de 1 celula)."""
        d = self.res
        dn = (self.height(n + d, e) - self.height(n - d, e)) / (2 * d)
        de = (self.height(n, e + d) - self.height(n, e - d)) / (2 * d)
        return np.arctan(np.hypot(dn, de))

    def segment_collision(self, p0, p1, step: float | None = None) -> bool:
        """True se o segmento p0->p1 (n, e, alt_msl) cruza o terreno.

        Evita 'tunelamento' quando o passo de tempo e grande.
        """
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        step = step or self.res / 2
        k = max(2, int(np.ceil(np.linalg.norm(p1[:2] - p0[:2]) / step)) + 1)
        s = np.linspace(0, 1, k)
        n = p0[0] + s * (p1[0] - p0[0])
        e = p0[1] + s * (p1[1] - p0[1])
        a = p0[2] + s * (p1[2] - p0[2])
        return bool(np.any(a < self.height(n, e)))

    def hillshade(self, azimuth=315.0, altitude=45.0):
        az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
        gy, gx = np.gradient(self.z, self.res)
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(gy, -gx)
        hs = (np.sin(alt) * np.cos(slope)
              + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
        return np.clip(hs, 0, 1)

    # ---------------------------------------------------------------- construtores
    @classmethod
    def synthetic(cls, size_km: float = 20.0, res: float = 30.0, relief: float = 600.0,
                  base: float = 900.0, beta: float = 2.4, seed: int = 0) -> "Terrain":
        """Terreno fractal (ruido 1/f^beta) para testes sem DEM real."""
        n = int(size_km * 1000 / res) + 1
        rng = np.random.default_rng(seed)
        fx = np.fft.fftfreq(n)[None, :]
        fy = np.fft.fftfreq(n)[:, None]
        f = np.hypot(fx, fy)
        f[0, 0] = 1.0
        spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / f ** (beta / 2 + 0.5)
        spec[0, 0] = 0
        z = np.fft.ifft2(spec).real
        z = (z - z.min()) / (z.max() - z.min())
        half = (n - 1) * res / 2
        return cls(base + relief * z, res, -half, -half)

    @classmethod
    def from_geotiffs(cls, paths, lat0: float, lon0: float, size_km: float = 20.0,
                      res: float = 30.0, cache_dir: str | None = None) -> "Terrain":
        """Recorta e reamostra tiles GeoTIFF (EPSG:4326) para a grade local NED."""
        half = size_km * 500.0
        tag = f"dem_{lat0:.4f}_{lon0:.4f}_{size_km:g}km_{res:g}m.npz"
        cpath = os.path.join(cache_dir, tag) if cache_dir else None
        if cpath and os.path.exists(cpath):
            d = np.load(cpath)
            return cls(d["z"], res, -half, -half, lat0, lon0)

        import rasterio
        from rasterio.merge import merge

        m_lon = M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
        dlat, dlon = (half + 200) / M_PER_DEG_LAT, (half + 200) / m_lon
        bounds = (lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat)
        srcs = [rasterio.open(p) for p in paths]
        try:
            mosaic, tr = merge(srcs, bounds=bounds, nodata=np.nan)
        finally:
            for s in srcs:
                s.close()
        g = mosaic[0].astype(np.float64)
        # preenche buracos (nodata) com a media para nao contaminar a interpolacao
        if np.isnan(g).any():
            g = np.where(np.isnan(g), np.nanmean(g), g)

        n = int(round(2 * half / res)) + 1
        nn = -half + res * np.arange(n)                  # norte [m]
        ee = -half + res * np.arange(n)                  # leste [m]
        lat = lat0 + nn / M_PER_DEG_LAT
        lon = lon0 + ee / m_lon
        # indices (fracionarios) no raster: centro de pixel em tr * (col+.5, row+.5)
        rows = (lat - tr.f) / tr.e - 0.5
        cols = (lon - tr.c) / tr.a - 0.5
        R, C = np.meshgrid(rows, cols, indexing="ij")
        z = map_coordinates(g, [R, C], order=1, mode="nearest")
        if cpath:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cpath, z=z)
        return cls(z, res, -half, -half, lat0, lon0)

    # -------------------------------------------------------------------- geodesia
    def to_latlon(self, n, e):
        lat0, lon0 = self.lat0 or 0.0, self.lon0 or 0.0
        return (lat0 + np.asarray(n) / M_PER_DEG_LAT,
                lon0 + np.asarray(e) / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))))
