"""Widget de mapa Tkinter: camadas (elevacao / satelite), zoom, pan e overlay desenhavel."""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from typing import Callable

import matplotlib
import numpy as np
from PIL import Image, ImageTk

from .terrain import Terrain
from .tiles import auto_zoom, fetch_satellite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_CACHE = os.path.join(ROOT, "data", "tiles")


def elevation_image(T: Terrain) -> Image.Image:
    z = T.z
    norm = matplotlib.colors.Normalize(z.min(), z.max())
    rgb = matplotlib.colormaps["terrain"](norm(z))[..., :3] * (0.6 + 0.4 * T.hillshade()[..., None])
    return Image.fromarray(np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[::-1]))


class MapView(tk.Canvas):
    """Canvas com mapa. Coordenadas do mundo: (n, e) em metros; pixels: (x, y)."""

    def __init__(self, parent, terrain: Terrain, on_click: Callable | None = None,
                 on_right: Callable | None = None, on_motion: Callable | None = None,
                 draw_overlay: Callable | None = None, on_press: Callable | None = None,
                 on_drag: Callable | None = None, on_release: Callable | None = None, **kw):
        super().__init__(parent, bg="#202020", highlightthickness=0, cursor="crosshair", **kw)
        self.on_click, self.on_right, self.on_motion = on_click, on_right, on_motion
        self.draw_overlay = draw_overlay
        # arrastar objetos: on_press(n, e) -> True captura o gesto (nao move o mapa); on_drag(n, e); on_release(moved)
        self.on_press, self.on_drag, self.on_release = on_press, on_drag, on_release
        self._grab = False
        self.layers: dict[str, Image.Image] = {}
        self.layer = "Elevacao"
        self.sat_state = "nao"          # nao | baixando | pronto | erro
        self.sat_msg = ""
        self._q: queue.Queue = queue.Queue()
        self.interactive = True
        self.set_terrain(terrain)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Button-3>", self._right)
        self.bind("<MouseWheel>", lambda e: self.zoom(e.x, e.y, 1.25 if e.delta > 0 else 0.8))
        self.bind("<Button-4>", lambda e: self.zoom(e.x, e.y, 1.25))
        self.bind("<Button-5>", lambda e: self.zoom(e.x, e.y, 0.8))
        self.bind("<Motion>", self._motion)
        self.bind("<Configure>", lambda e: self.redraw())

    # ------------------------------------------------------------------ terreno
    def set_terrain(self, terrain: Terrain):
        self.T = terrain
        self.layers = {"Elevacao": elevation_image(terrain)}
        self.layer = "Elevacao"
        self.sat_state, self.sat_msg = "nao", ""
        self.fit()

    def fit(self):
        T = self.T
        W, H = max(self.winfo_width(), 200), max(self.winfo_height(), 200)
        self.cn, self.ce = (T.n0 + T.n_max) / 2, (T.e0 + T.e_max) / 2
        self.scale = min(W / (T.e_max - T.e0), H / (T.n_max - T.n0))        # px por metro

    def center_on(self, n, e):
        self.cn, self.ce = n, e

    # ------------------------------------------------------------- transformacoes
    def w2s(self, n, e):
        return (self.winfo_width() / 2 + (e - self.ce) * self.scale,
                self.winfo_height() / 2 - (n - self.cn) * self.scale)

    def s2w(self, x, y):
        return (self.cn - (y - self.winfo_height() / 2) / self.scale,
                self.ce + (x - self.winfo_width() / 2) / self.scale)

    # ------------------------------------------------------------------- camadas
    def set_layer(self, name: str):
        if name == "Satelite" and "Satelite" not in self.layers:
            self.request_satellite()
            return
        if name in self.layers:
            self.layer = name
            self.redraw()

    def request_satellite(self, zoom: int | None = None):
        if self.sat_state == "baixando":
            return
        if self.T.lat0 is None:
            self.sat_state, self.sat_msg = "erro", "terreno sintetico: sem georreferencia"
            return
        self.sat_state = "baixando"
        z = zoom or auto_zoom(self.T)
        self.sat_msg = f"baixando satelite (zoom {z})..."

        def work():
            try:
                img = fetch_satellite(self.T, TILE_CACHE, z,
                                      progress=lambda d, t: self._q.put(("p", d, t)))
                self._q.put(("ok", img))
            except Exception as ex:                              # sem rede, etc.
                self._q.put(("err", str(ex)))
        threading.Thread(target=work, daemon=True).start()
        self.after(200, self._poll_sat)

    def _poll_sat(self):
        try:
            while True:
                m = self._q.get_nowait()
                if m[0] == "p":
                    self.sat_msg = f"baixando satelite: {m[1]}/{m[2]} tiles"
                elif m[0] == "ok":
                    self.layers["Satelite"] = m[1]
                    self.layer, self.sat_state, self.sat_msg = "Satelite", "pronto", "satelite carregado"
                    self.redraw()
                    return
                else:
                    self.sat_state, self.sat_msg = "erro", f"satelite falhou: {m[1]}"
                    self.layer = "Elevacao"
                    return
        except queue.Empty:
            pass
        self.after(200, self._poll_sat)

    # -------------------------------------------------------------------- eventos
    def _press(self, ev):
        self._p0, self._pan0, self._moved = (ev.x, ev.y), (self.cn, self.ce), False
        self._grab = bool(self.on_press and self.on_press(*self.s2w(ev.x, ev.y)))

    def _drag(self, ev):
        dx, dy = ev.x - self._p0[0], ev.y - self._p0[1]
        if not self._moved and abs(dx) + abs(dy) < 5:
            return
        self._moved = True
        if self._grab:
            if self.on_drag:
                self.on_drag(*self.s2w(ev.x, ev.y))
            return
        if self.interactive:
            self.cn = self._pan0[0] + dy / self.scale
            self.ce = self._pan0[1] - dx / self.scale
            self.redraw(light=True)

    def _release(self, ev):
        if self._grab:
            self._grab = False
            if self.on_release:
                self.on_release(self._moved)
        elif self._moved:
            self.redraw()
        elif self.on_click:
            self.on_click(*self.s2w(ev.x, ev.y))

    def _right(self, ev):
        if self.on_right:
            self.on_right(*self.s2w(ev.x, ev.y))

    def _motion(self, ev):
        if self.on_motion:
            self.on_motion(*self.s2w(ev.x, ev.y))

    def zoom(self, x, y, f):
        n, e = self.s2w(x, y)
        self.scale = float(np.clip(self.scale * f, 0.003, 3.0))
        self.cn = n + (y - self.winfo_height() / 2) / self.scale
        self.ce = e - (x - self.winfo_width() / 2) / self.scale
        self.redraw()

    # --------------------------------------------------------------------- desenho
    def _render_bg(self, W, H):
        T, img = self.T, self.layers[self.layer]
        n_top, e_left = self.s2w(0, 0)
        n_bot, e_right = self.s2w(W, H)
        kx, ky = img.width / (T.e_max - T.e0), img.height / (T.n_max - T.n0)
        l, r = (e_left - T.e0) * kx, (e_right - T.e0) * kx
        t, b = (T.n_max - n_top) * ky, (T.n_max - n_bot) * ky
        out = Image.new("RGB", (W, H), (32, 32, 32))
        l2, r2, t2, b2 = max(l, 0), min(r, img.width), max(t, 0), min(b, img.height)
        if r2 - l2 > 1 and b2 - t2 > 1:
            x0, x1 = (l2 - l) / (r - l) * W, (r2 - l) / (r - l) * W
            y0, y1 = (t2 - t) / (b - t) * H, (b2 - t) / (b - t) * H
            w, h = max(int(round(x1 - x0)), 1), max(int(round(y1 - y0)), 1)
            crop = img.resize((w, h), Image.BILINEAR, box=(l2, t2, r2, b2), reducing_gap=2.0)
            out.paste(crop, (int(round(x0)), int(round(y0))))
        return out

    def redraw(self, light: bool = False):
        W, H = self.winfo_width(), self.winfo_height()
        if W < 10 or H < 10:
            return
        self._tk_img = ImageTk.PhotoImage(self._render_bg(W, H))
        self.delete("all")
        self.create_image(0, 0, image=self._tk_img, anchor="nw")
        if self.draw_overlay:
            self.draw_overlay(self, light)
