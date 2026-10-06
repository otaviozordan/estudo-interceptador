"""Replay interativo (Tkinter) de uma simulacao: mapa com a bolinha, trajetoria real x teorica,
comandos dos atuadores e parametros de voo em tempo (quase) real, com satelite/elevacao.

Teclas: espaco = play/pause | setas = +-1 s | F = seguir aeronave | R = reiniciar
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from .landmarks import draw_landmarks, visible_landmarks
from .mapview import MapView
from .simulation import Simulation
from .theme import BG, FG, LINE, MUTED, PANEL2, WARN, apply_dark

COLORS = ["#4da3ff", "#ff5a5a", "#4cd964", "#c08cff", "#ffb04d", "#4de0f0"]   # linhas sobre fundo escuro
BRIGHT = ["#4da3ff", "#ff5a5a", "#4cd964", "#c08cff", "#ffb04d", "#4de0f0"]   # sobre mapa escuro
MODE_NAMES = {0: "segmento (L1)", 1: "orbita", 2: "orbita final"}
FOLLOW_SCALE = 0.15   # px/m ao seguir (~7 km de largura)
SURF_LIM = 30.0   # graus (limite dos atuadores)


class ReplayApp:
    def __init__(self, root: tk.Tk, sim: Simulation, title: str = "Replay da simulacao"):
        self.root, self.sim, self.T = root, sim, sim.terrain
        self.res = sim.results()
        self.names = [a.name for a in sim.agents if len(self.res[a.name]["t"])]
        self.agents = {a.name: a for a in sim.agents}
        self.dt = sim.dt
        self.t_end = max(self.res[n]["t"][-1] for n in self.names)
        self.t = 0.0
        self.playing = False
        self.speed = 1.0
        self.cur = self.names[0]
        self._last = time.perf_counter()
        self._guard = False
        root.title(title)
        apply_dark(root)
        root.geometry(f"{min(1400, root.winfo_screenwidth() - 40)}x{min(900, root.winfo_screenheight() - 90)}+10+10")
        self._build_ui()
        root.bind("<space>", lambda e: self.toggle())
        root.bind("<Left>", lambda e: self.seek(self.t - 1))
        root.bind("<Right>", lambda e: self.seek(self.t + 1))
        root.bind("f", lambda e: (self.follow_var.set(not self.follow_var.get()), self._follow_toggled()))
        root.bind("r", lambda e: self.seek(0))
        root.update_idletasks()
        self.map.fit()
        self.map.scale = max(self.map.scale, FOLLOW_SCALE)      # abre ja acompanhando a bolinha
        self._select_changed()
        self.update_frame()
        self.layer_var.set("Satelite")             # camada padrao; sem rede volta ao relevo (ver _tick)
        self.map.set_layer("Satelite")
        self.root.after(30, self._tick)

    # --------------------------------------------------------------------- UI
    def _build_ui(self):
        root = self.root
        bar = ttk.Frame(root, padding=4)
        bar.pack(side="top", fill="x")
        self.play_btn = ttk.Button(bar, text="Play", width=7, command=self.toggle)
        self.play_btn.pack(side="left")
        ttk.Button(bar, text="Reiniciar", command=lambda: self.seek(0)).pack(side="left", padx=2)
        ttk.Label(bar, text=" vel.").pack(side="left")
        self.speed_cb = ttk.Combobox(bar, values=["0.25x", "0.5x", "1x", "2x", "4x", "8x"], width=5,
                                     state="readonly")
        self.speed_cb.set("1x")
        self.speed_cb.pack(side="left")
        self.speed_cb.bind("<<ComboboxSelected>>",
                           lambda e: setattr(self, "speed", float(self.speed_cb.get()[:-1])))
        self.slider = ttk.Scale(bar, from_=0, to=self.t_end, command=self._slider_moved)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.time_lbl = ttk.Label(bar, text="0.0 / 0.0 s", width=14)
        self.time_lbl.pack(side="left")
        self.follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Seguir", variable=self.follow_var,
                        command=self._follow_toggled).pack(side="left", padx=4)
        self.trail_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Rastro", variable=self.trail_var, command=self.update_frame).pack(side="left")
        self.net = self.sim.radar_net
        self.radar_var = tk.BooleanVar(value=True)
        self.radar_only = tk.BooleanVar(value=False)      # esconde a verdade: so radares, plots e pistas
        if self.net:
            ttk.Checkbutton(bar, text="Radar", variable=self.radar_var, command=self.update_frame).pack(side="left")
            ttk.Checkbutton(bar, text="Visão radar", variable=self.radar_only, command=self._radar_only_toggled
                            ).pack(side="left", padx=4)
        self.layer_var = tk.StringVar(value="Elevacao")
        for name in ("Elevacao", "Satelite"):
            ttk.Radiobutton(bar, text=name, value=name, variable=self.layer_var,
                            command=lambda: (self.map.set_layer(self.layer_var.get()), self.update_frame())
                            ).pack(side="left", padx=2)
        self.status = ttk.Label(bar, text="", width=28)
        self.status.pack(side="left")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        side = ttk.Frame(body, padding=8, width=300)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True)
        self.map = MapView(center, self.T, draw_overlay=self._overlay, on_motion=self._on_motion)
        self.map.pack(fill="both", expand=True)
        self.strips = tk.Canvas(center, height=190, bg=PANEL2, highlightthickness=0)
        self.strips.pack(fill="x")
        self.strips.bind("<Configure>", lambda e: self._select_changed())
        self.strips.bind("<Button-1>", self._strip_click)
        self.info = ttk.Label(center, text="", anchor="w")
        self.info.pack(fill="x")

        ttk.Label(side, text="Aeronave", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.ac_cb = ttk.Combobox(side, values=self.names, state="readonly")
        self.ac_cb.set(self.cur)
        self.ac_cb.pack(fill="x")
        self.ac_cb.bind("<<ComboboxSelected>>", lambda e: self._set_cur(self.ac_cb.get()))
        self.state_lbl = ttk.Label(side, text="", foreground=WARN, font=("Segoe UI", 10, "bold"))
        self.state_lbl.pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Parametros de voo", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        grid = ttk.Frame(side)
        grid.pack(fill="x")
        self.tel: dict[str, tk.StringVar] = {}
        rows = [("pos", "Posicao N/E [m]"), ("alt", "Altitude MSL [m]"), ("terr", "Terreno [m]"),
                ("agl", "AGL [m]"), ("va", "Va (cmd) [m/s]"), ("vg", "Vg [m/s]"),
                ("vz", "Vz (cmd) [m/s]"), ("phi", "Rolagem (cmd) [deg]"), ("theta", "Arfagem [deg]"),
                ("psi", "Rumo [deg]"), ("alpha", "Alpha [deg]"), ("pqr", "p/q/r [deg/s]"),
                ("xt", "Erro de rota [m]"), ("wp", "Waypoint / modo")]
        for i, (k, label) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value="-")
            ttk.Label(grid, textvariable=v, font=("Consolas", 10)).grid(row=i, column=1, sticky="e", padx=(8, 0))
            self.tel[k] = v
        grid.columnconfigure(1, weight=1)

        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Comandos (barra = comando, traco = atuador)",
                  font=("Segoe UI", 10, "bold"), wraplength=280).pack(anchor="w")
        self.bars = tk.Canvas(side, height=150, bg=PANEL2, highlightthickness=0)
        self.bars.pack(fill="x", pady=4)
        if self.net:
            ttk.Separator(side).pack(fill="x", pady=6)
            ttk.Label(side, text="Radar (unica fonte de posicao)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            self.radar_lbl = ttk.Label(side, text="", justify="left", font=("Consolas", 9), wraplength=280)
            self.radar_lbl.pack(anchor="w")
            ttk.Label(side, text="Parametros do radar", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            self.radar_cb = ttk.Combobox(side, values=[r.name for r in self.net.radars], state="readonly")
            self.radar_cb.set(self.net.radars[0].name)
            self.radar_cb.pack(fill="x")
            self.radar_cb.bind("<<ComboboxSelected>>", lambda e: self.update_frame())
            self.radar_info = ttk.Label(side, text="", justify="left", font=("Consolas", 9))
            self.radar_info.pack(anchor="w")
        ttk.Label(side, text="Legenda do mapa:\n  linha tracejada = rota teorica (waypoints)\n"
                             "  linha cheia = trajetoria real\n  bolinha = aeronave\n"
                             "  triangulo laranja = radar | + = deteccao (ruidosa)\n"
                             "  quadrado laranja = pista estimada (seta = velocidade)",
                  justify="left").pack(anchor="w", pady=6)

    # ----------------------------------------------------------------------- dados
    def _idx(self, name: str, t: float | None = None) -> int:
        r = self.res[name]["t"]
        i = int(round((self.t if t is None else t) / self.dt))
        return max(0, min(i, len(r) - 1))

    def _set_cur(self, name):
        self.cur = name
        self._select_changed()
        self.update_frame()

    # -------------------------------------------------------------------- controle
    def _follow_toggled(self):
        if self.follow_var.get():
            self.map.scale = max(self.map.scale, FOLLOW_SCALE)
        else:
            self.map.fit()
        self.update_frame()

    def toggle(self):
        if not self.playing and self.t >= self.t_end - 1e-6:
            self.t = 0.0
        self.playing = not self.playing
        self.play_btn.config(text="Pausa" if self.playing else "Play")
        self._last = time.perf_counter()

    def seek(self, t):
        self.t = float(np.clip(t, 0, self.t_end))
        self.update_frame()

    def _slider_moved(self, v):
        if self._guard:
            return
        self.t = float(v)
        self.update_frame(from_slider=True)

    def _strip_click(self, ev):
        for ox, ox1, _a, _b in self._panel_geo:
            if ox <= ev.x <= ox1:
                self.seek((ev.x - ox) / (ox1 - ox) * self.t_end)

    def _tick(self):
        now = time.perf_counter()
        if self.playing:
            self.t += (now - self._last) * self.speed
            if self.t >= self.t_end:
                self.t = self.t_end
                self.playing = False
                self.play_btn.config(text="Play")
            self.update_frame()
        self._last = now
        s = self.map
        self.status.config(text=s.sat_msg if s.sat_state in ("baixando", "erro") else "")
        if s.sat_state == "erro":
            self.layer_var.set("Elevacao")
            s.sat_state = "nao"
        self.root.after(30, self._tick)

    def _on_motion(self, n, e):
        if self.T.inside(n, e):
            lat, lon = self.T.to_latlon(n, e)
            self.info.config(text=f"N={n:8.0f}  E={e:8.0f}  elev={float(self.T.height(n, e)):5.0f} m   "
                                  f"lat={float(lat):.5f}  lon={float(lon):.5f}")

    # ------------------------------------------------------------------ atualizacao
    def update_frame(self, from_slider: bool = False):
        if not from_slider:
            self._guard = True
            self.slider.set(self.t)
            self._guard = False
        self.time_lbl.config(text=f"{self.t:5.1f} / {self.t_end:.1f} s")
        r = self.res[self.cur]
        i = self._idx(self.cur)
        if self.follow_var.get() and not self._radar_view():
            self.map.center_on(r["n"][i], r["e"][i])
        self.map.redraw()
        self._telemetry(r, i)
        self._bars(r, i)
        if self.net:
            self.radar_info.config(text=self._radar_info())
        self._move_cursors()

    def _draw_gusts(self, c: MapView):
        """Zonas de rajada (circulo + seta); ficam em destaque enquanto alguma aeronave esta dentro."""
        for k, g in enumerate(self.sim.gusts):
            inside = False
            for name in self.names:
                r, i = self.res[name], self._idx(name)
                inside = inside or g.weight(r["n"][i], r["e"][i]) > 0
            col = "#ffe14d" if inside else "#5ad1ff"
            x, y = c.w2s(g.n, g.e)
            R = g.radius * c.scale
            c.create_oval(x - R, y - R, x + R, y + R, outline=col, width=3 if inside else 2, dash=(7, 4))
            c.create_oval(x - R / 2, y - R / 2, x + R / 2, y + R / 2, outline=col, dash=(2, 4))
            h, L = np.radians(g.dir_deg), max(0.7 * R, 22.0)
            c.create_line(x, y, x + L * np.sin(h), y - L * np.cos(h), fill=col, width=3, arrow="last",
                          arrowshape=(12, 14, 5))
            c.create_text(x, y - max(R, 14) - 6, anchor="s", fill=col, font=("Segoe UI", 9, "bold"),
                          text=f"rajada {k + 1}: {g.speed:g} m/s → {g.dir_deg:03.0f}°" + ("  (ATIVA)" if inside else ""))

    def _radar_only_toggled(self):
        if self.radar_only.get():
            self.radar_var.set(True)
        self._select_changed()
        self.update_frame()

    def _draw_radar(self, c: MapView):
        """Estacoes (alcance), deteccoes recentes (+) e pistas estimadas (quadrado + velocidade + rastro)."""
        col = "#ff9f43"
        for rd in self.net.radars:
            x, y = c.w2s(rd.n, rd.e)
            R = rd.range_m * c.scale
            if rd.az_width_deg >= 360.0:
                c.create_oval(x - R, y - R, x + R, y + R, outline=col, dash=(8, 5))
            else:
                c.create_arc(x - R, y - R, x + R, y + R, start=90.0 - (rd.az_center_deg + rd.az_width_deg / 2),
                             extent=rd.az_width_deg, style="arc", outline=col)
            c.create_polygon(x, y - 10, x + 9, y + 6, x - 9, y + 6, fill=col, outline="black", width=2)
            if rd.name == self.radar_cb.get():
                c.create_oval(x - 15, y - 15, x + 15, y + 15, outline="yellow", width=2)
            c.create_text(x + 12, y - 8, text=rd.name, fill="white", anchor="sw", font=("Segoe UI", 9, "bold"))
        span = 1.5 * max(r.scan_period for r in self.net.radars)
        for d in self.net.detections_between(self.t - span, self.t):
            x, y = c.w2s(d.z[0], d.z[1])
            c.create_line(x - 4, y, x + 4, y, fill="#ffd9a8", width=2)
            c.create_line(x, y - 4, x, y + 4, fill="#ffd9a8", width=2)
        for tid, log in self.net.track_log.items():
            pts = [(smp.x[0], smp.x[1]) for smp in log if self.t - 90.0 <= smp.t <= self.t and smp.confirmed]
            if len(pts) >= 2:
                c.create_line(*[v for p in pts for v in c.w2s(*p)], fill=col, width=1)
        for trk in self.net.picture(self.t):
            x, y = c.w2s(trk["n"], trk["e"])
            c.create_line(x, y, x + 15 * trk["ve"] * c.scale, y - 15 * trk["vn"] * c.scale, fill=col, width=2,
                          arrow="last")
            c.create_rectangle(x - 6, y - 6, x + 6, y + 6, outline="black", width=4)
            c.create_rectangle(x - 6, y - 6, x + 6, y + 6, outline=col, width=2)
            c.create_text(x + 9, y + 8, text=f"T{trk['id']} +-{trk['sigma']:.0f} m", fill=col, anchor="nw",
                          font=("Segoe UI", 8, "bold"))

    def _radar_text(self) -> str:
        if self._radar_view():
            pic = sorted(self.net.picture(self.t), key=lambda p: p["id"])
            lines = [f"pistas ativas: {len(pic)}", "ID   N/E [m]        alt  vel  rumo  +-m"]
            for p in pic:
                brg = np.degrees(np.arctan2(p["ve"], p["vn"])) % 360
                lines.append(f"T{p['id']:<3}{p['n']:7.0f}/{p['e']:<7.0f}{p['alt']:5.0f}{np.hypot(p['vn'], p['ve']):5.0f}"
                             f"{brg:5.0f}{p['sigma']:5.0f}")
            return "\n".join(lines)
        i = self._idx(self.cur)
        r = self.res[self.cur]
        truth = np.array([r["n"][i], r["e"][i], r["alt"][i]])
        mine = [p for p in self.net.picture(self.t) if p["truth"] == self.cur]
        lines = [f"pistas ativas: {len(self.net.picture(self.t))}"]
        if mine:
            p = min(mine, key=lambda q: np.linalg.norm(np.array([q["n"], q["e"], q["alt"]]) - truth))
            err = float(np.linalg.norm(np.array([p["n"], p["e"], p["alt"]]) - truth))
            lines += [f"{self.cur}: T{p['id']}  erro 3D {err:.0f} m", f"  incerteza +-{p['sigma']:.0f} m  "
                      f"idade {p['age']:.1f} s", f"  vel. est. {np.hypot(p['vn'], p['ve']):.0f} m/s"]
        else:
            lines.append(f"{self.cur}: SEM PISTA")
        return "\n".join(lines)

    def _overlay(self, c: MapView, light: bool):
        t_cur = self.t
        if not hasattr(self, "_landmarks"):
            self._landmarks = visible_landmarks(self.T)
        draw_landmarks(c, self._landmarks)                  # marcos (mapa: valem tambem na visao radar)
        if self.net and self.radar_only.get():          # visao do operador: nada da verdade-terreno
            self._draw_radar(c)
            return
        self._draw_gusts(c)
        if self.net and self.radar_var.get():
            self._draw_radar(c)
        for k, name in enumerate(self.names):
            col, bright = COLORS[k % 6], BRIGHT[k % 6]
            a, r = self.agents[name], self.res[name]
            i = self._idx(name)
            # --- trajetoria teorica: rota de waypoints (+ orbitas)
            wps = a.navigator.wps
            pts = [(r["n"][0], r["e"][0])] + [(w.n, w.e) for w in wps]
            if a.navigator.route.loop and len(wps) > 2:
                pts.append((wps[0].n, wps[0].e))
            c.create_line(*[v for p in pts for v in c.w2s(*p)], fill="white", width=3, dash=(8, 6))
            c.create_line(*[v for p in pts for v in c.w2s(*p)], fill=bright, width=1, dash=(8, 6))
            for j, w in enumerate(wps):
                x, y = c.w2s(w.n, w.e)
                if w.impact:
                    c.create_line(x - 8, y - 8, x + 8, y + 8, fill="black", width=5)
                    c.create_line(x - 8, y + 8, x + 8, y - 8, fill="black", width=5)
                    c.create_line(x - 7, y - 7, x + 7, y + 7, fill=bright, width=2)
                    c.create_line(x - 7, y + 7, x + 7, y - 7, fill=bright, width=2)
                else:
                    c.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill="white", outline=col, width=2)
                if w.loiter_radius > 0:
                    rr = w.loiter_radius * c.scale
                    c.create_oval(x - rr, y - rr, x + rr, y + rr, outline=bright, dash=(4, 4), width=1)
            # --- trajetoria real
            if self.trail_var.get() and i > 1:
                step = max(1, i // 1500)
                n, e = r["n"][:i + 1:step], r["e"][:i + 1:step]
                xy = [v for q in zip(n, e) for v in c.w2s(*q)]
                if len(xy) >= 4:
                    c.create_line(*xy, fill="black", width=5)
                    c.create_line(*xy, fill=bright, width=3)
            # --- bolinha
            x, y = c.w2s(r["n"][i], r["e"][i])
            sel = name == self.cur
            rad = 9 if sel else 7
            alive = not (a.crash_time is not None and t_cur >= a.crash_time - 1e-6)
            if alive:
                psi = r["psi"][i]
                c.create_line(x, y, x + 22 * np.sin(psi), y - 22 * np.cos(psi), fill="white", width=3)
                c.create_oval(x - rad - 3, y - rad - 3, x + rad + 3, y + rad + 3,
                              outline="white" if sel else "", width=2)
                c.create_oval(x - rad, y - rad, x + rad, y + rad, fill=col, outline="white", width=2)
            else:
                c.create_oval(x - 14, y - 14, x + 14, y + 14, fill="#ff3000", outline="white", width=2)
                c.create_text(x, y, text="X", fill="white", font=("Segoe UI", 12, "bold"))
            c.create_text(x + rad + 6, y - rad - 4, text=name + ("" if alive else " (colidiu)"),
                          fill="white", anchor="sw", font=("Segoe UI", 10, "bold"))

    def _sel_radar(self) -> int:
        return self.net.radars.index(next(r for r in self.net.radars if r.name == self.radar_cb.get()))

    def _radar_info(self) -> str:
        """Parametros internos do radar escolhido + estado da ultima varredura (e geometria ate a aeronave)."""
        k = self._sel_radar()
        rd, net = self.net.radars[k], self.net
        T = self.T
        L = [f"posicao N/E   {rd.n:.0f} / {rd.e:.0f} m",
             f"terreno/antena {float(T.height(rd.n, rd.e)):.0f} / {net._ant[k]:.0f} m MSL (+{rd.height:g})",
             f"alcance 1 m2  {rd.range_m / 1000:g} km (Pd50, visada livre)",
             f"zona cega     {rd.min_range:g} m",
             f"varredura     {rd.scan_period:g} s",
             f"sigma r/az/el {rd.range_sigma:g} m / {rd.az_sigma_deg:g} / {rd.el_sigma_deg:g} deg",
             f"frequencia    {rd.frequency_ghz:g} GHz  atm {rd.atm_loss_db_km:g} dB/km",
             f"vies r/az     {rd.range_bias:+g} m / {rd.az_bias_deg:+g} deg",
             f"SNR50 / Pfa   {10 * np.log10(rd.snr50):.1f} dB / {rd.pfa:g}",
             f"elevacao      {rd.min_elev_deg:g}..{rd.max_elev_deg:g} deg",
             "setor         " + ("360 (total)" if rd.az_width_deg >= 360 else
                                 f"{rd.az_center_deg:g} +-{rd.az_width_deg / 2:g} deg"),
             f"falsos alarmes {rd.false_alarms:g}/varredura",
             f"estado        {'ligado' if rd.enabled else 'DESLIGADO'}"]
        scans = [x for x in net.scan_log if x[1] == k and x[0] <= self.t]
        dets = sum(x[2] for x in scans)
        L.append(f"varreduras    {len(scans)}  (plots {dets})")
        if scans:
            t, _k, nd, nv = scans[-1]
            L.append(f"ultima        t={t:.1f}s: {nd} plot(s)" + ("" if self._radar_view() else f", {nv} visivel(is)"))
        if not self._radar_view():                       # geometria ate a aeronave: usa a verdade
            r, i = self.res[self.cur], self._idx(self.cur)
            tg = np.array([r["n"][i], r["e"][i], r["alt"][i]])
            lk = net.link(k, tg, float(r["psi"][i]), self.agents[self.cur].rcs)
            L += ["", f"-> {self.cur}:", f"  dist {lk['r'] / 1000:.2f} km  az {lk['az']:.0f}  el {lk['el']:+.1f}",
                  f"  RCS efetiva {lk['rcs_eff']:.2f} m2 (base {self.agents[self.cur].rcs:g})",
                  f"  perda do relevo {lk['loss_db']:.1f} dB",
                  f"  SNR {10 * np.log10(max(lk['snr'], 1e-9)):+.1f} dB  Pd {lk['pd']:.2f}"
                  + ("" if lk["in_beam"] else "  (fora do feixe)")]
        return "\n".join(L)

    def _radar_view(self) -> bool:
        return bool(self.net) and self.radar_only.get()

    def _telemetry(self, r, i):
        if self._radar_view():
            for v in self.tel.values():
                v.set("-")
            self.state_lbl.config(text="")
            self.radar_lbl.config(text=self._radar_text())
            return
        a = self.agents[self.cur]
        d = np.degrees
        self.tel["pos"].set(f"{r['n'][i]:8.0f} /{r['e'][i]:8.0f}")
        self.tel["alt"].set(f"{r['alt'][i]:7.1f}  (cmd {r['alt_cmd'][i]:.0f})")
        self.tel["terr"].set(f"{r['terrain'][i]:7.1f}")
        self.tel["agl"].set(f"{r['agl'][i]:7.1f}")
        self.tel["va"].set(f"{r['Va'][i]:5.1f} ({r['speed_cmd'][i]:.0f})")
        self.tel["vg"].set(f"{r['Vg'][i]:5.1f}")
        self.tel["vz"].set(f"{r['hdot'][i]:+5.1f} ({r['hdot_cmd'][i]:+.1f})")
        self.tel["phi"].set(f"{d(r['phi'][i]):+6.1f} ({d(r['phi_cmd'][i]):+.0f})")
        self.tel["theta"].set(f"{d(r['theta'][i]):+6.1f}")
        self.tel["psi"].set(f"{d(r['psi'][i]) % 360:6.1f}")
        self.tel["alpha"].set(f"{d(r['alpha'][i]):+6.1f}")
        self.tel["pqr"].set(f"{d(r['p'][i]):+5.0f}/{d(r['q'][i]):+4.0f}/{d(r['r'][i]):+4.0f}")
        self.tel["xt"].set(f"{r['xtrack'][i]:+7.1f}")
        self.tel["wp"].set(f"{int(r['wp'][i]) + 1} / {MODE_NAMES.get(int(r['mode'][i]), '?')}")
        crashed = a.crash_time is not None and self.t >= a.crash_time - 1e-6
        if self.net:
            self.radar_lbl.config(text=self._radar_text())
        self.state_lbl.config(text=f"COLISAO ({a.crash_cause}) em t={a.crash_time:.1f}s" if crashed else "")

    def _bars(self, r, i):
        b = self.bars
        b.delete("all")
        if self._radar_view():
            return
        W = max(b.winfo_width(), 200)
        items = [("Profundor (de)", "de", "de_c", -SURF_LIM, SURF_LIM, True),
                 ("Aileron (da)", "da", "da_c", -SURF_LIM, SURF_LIM, True),
                 ("Leme (dr)", "dr", "dr_c", -SURF_LIM, SURF_LIM, True),
                 ("Manete (dt)", "dt", "dt_c", 0, 100, False)]
        x0, x1 = 100, W - 70
        for k, (label, act, cmd, lo, hi, deg) in enumerate(items):
            y = 18 + k * 34
            cv, av = r[cmd][i], r[act][i]
            if deg:
                cv, av = np.degrees(cv), np.degrees(av)
            else:
                cv, av = cv * 100, av * 100
            cv, av = float(np.clip(cv, lo, hi)), float(np.clip(av, lo, hi))
            X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
            b.create_text(4, y, text=label, anchor="w", fill=FG, font=("Segoe UI", 9))
            b.create_rectangle(x0, y - 8, x1, y + 8, outline=LINE, fill=BG)
            zero = X(0 if deg else lo)
            b.create_rectangle(min(zero, X(cv)), y - 6, max(zero, X(cv)), y + 6, fill=BRIGHT[0], outline="")
            b.create_line(X(av), y - 10, X(av), y + 10, fill="#d02000", width=3)
            b.create_text(x1 + 6, y, text=f"{cv:+.1f}" if deg else f"{cv:.0f}%", anchor="w",
                          fill=FG, font=("Consolas", 9))
            if deg:
                b.create_line(X(0), y - 10, X(0), y + 10, fill=MUTED)

    # ---------------------------------------------------------------- series temporais
    def _select_changed(self):
        """Redesenha os 4 graficos (real x comando) da aeronave selecionada."""
        c = self.strips
        c.delete("all")
        W, H = max(c.winfo_width(), 400), c.winfo_height()
        if self._radar_view():
            self._panel_geo = []
            c.create_text(W / 2, H / 2, fill=MUTED, font=("Segoe UI", 11),
                          text="Visão radar: só medidas e pistas estimadas (sem telemetria da aeronave)")
            return
        r = self.res[self.cur]
        t = r["t"]
        d = np.degrees
        panels = [("Altitude MSL [m]", [(r["terrain"], "#a07850", "fill"), (r["alt_cmd"], "#9aa6b8", "dash"),
                                        (r["alt"], COLORS[0], "line")]),
                  ("AGL [m]", [(r["agl"], COLORS[2], "line")]),
                  ("Va [m/s]  (real x cmd)", [(r["speed_cmd"], "#9aa6b8", "dash"), (r["Va"], COLORS[1], "line")]),
                  ("Rolagem [deg]  (real x cmd)", [(d(r["phi_cmd"]), "#9aa6b8", "dash"), (d(r["phi"]), COLORS[3], "line")])]
        if self.net:
            te, ee = self.net.error_series(self.cur)
            if len(te):
                panels.append(("Erro de posicao do radar [m]", [(ee, COLORS[4], "line", te)]))
        pw = W / len(panels)
        self._panel_geo = []
        for k, (title, series) in enumerate(panels):
            ox, ox1 = k * pw + 40, (k + 1) * pw - 10
            oy, oy1 = 22, H - 20
            vals = np.concatenate([s[0] for s in series])
            lo, hi = float(np.min(vals)), float(np.max(vals))
            if title.startswith("AGL"):
                lo = min(lo, 0.0)
            if hi - lo < 1e-6:
                hi = lo + 1
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            X = lambda tt, ox=ox, ox1=ox1: ox + tt / max(self.t_end, 1e-6) * (ox1 - ox)
            Y = lambda v, lo=lo, hi=hi, oy=oy, oy1=oy1: oy1 - (v - lo) / (hi - lo) * (oy1 - oy)
            c.create_rectangle(ox, oy, ox1, oy1, outline=LINE)
            c.create_text(ox, 4, text=title, anchor="nw", fill=FG, font=("Segoe UI", 9, "bold"))
            c.create_text(ox - 3, oy, text=f"{hi:.0f}", anchor="e", fill=MUTED, font=("Consolas", 8))
            c.create_text(ox - 3, oy1, text=f"{lo:.0f}", anchor="e", fill=MUTED, font=("Consolas", 8))
            step = max(1, len(t) // 400)
            for arr, col, kind, *xs in series:
                tx = xs[0] if xs else t
                st = max(1, len(tx) // 400)
                if kind == "fill":
                    poly = [(ox, oy1)] + [(X(tt), Y(v)) for tt, v in zip(tx[::st], arr[::st])] + [(X(tx[-1]), oy1)]
                    c.create_polygon([q for p in poly for q in p], fill=col, outline="")
                else:
                    pts = [q for tt, v in zip(tx[::st], arr[::st]) for q in (X(tt), Y(v))]
                    if len(pts) >= 4:
                        c.create_line(*pts, fill=col, width=2 if kind == "line" else 1,
                                      dash=(4, 3) if kind == "dash" else ())
            self._panel_geo.append((ox, ox1, oy, oy1))
        self._move_cursors()

    def _move_cursors(self):
        c = self.strips
        c.delete("cursor")
        for ox, ox1, oy, oy1 in getattr(self, "_panel_geo", []):
            x = ox + self.t / max(self.t_end, 1e-6) * (ox1 - ox)
            c.create_line(x, oy, x, oy1, fill="white", tags="cursor")


def play(sim: Simulation, title: str = "Replay da simulacao"):
    root = tk.Tk()
    ReplayApp(root, sim, title)
    root.mainloop()
