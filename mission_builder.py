"""Construtor grafico de missao (Tkinter) com foco em estudo de controle de trafego aereo.

Uso:
    python mission_builder.py [missions/minha.json] [--place itajuba | --lat -23.03 --lon -45.55 --size 12] [--synthetic]

Mapa:   clique esquerdo = adiciona (conforme o modo)  |  arrastar objeto = move (waypoint, inicio, fixo, pista)
        arrastar o fundo = mover o mapa  |  scroll = zoom  |  clique direito = remove o objeto mais proximo
Modos:  1 Waypoint | 2 Chao (impacto) | 3 Loiter (orbita/espera) | 4 Inicio | 5 Rajada | 6 Radar
Atalhos: Ctrl+S salva, Ctrl+Z desfaz, Del remove o waypoint selecionado, Esc limpa selecao.
Rota suave: marque 'Rota suave (spline)' na aba Aeronaves (Catmull-Rom passando por todos os waypoints).
"""
from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from swarm_sim.mapview import MapView
from swarm_sim.gusts import GustZone
from swarm_sim.radar import RadarStation, coverage_mask
from swarm_sim.mission import PLACES, AircraftSpec, Mission, load_terrain, place_origin
from swarm_sim.routes import Waypoint
from swarm_sim.theme import FG, LINE, MUTED, PANEL2, apply_dark

# paleta
COLORS = ["#4aa3ff", "#ff6b6b", "#4cd98a", "#c792ea", "#ffb454", "#2ee6d6", "#f4e04d", "#ff8fd1"]
MODES = [("Waypoint", "➤ Waypoint"), ("Chao", "✖ Chão"), ("Loiter", "⟳ Loiter"),
         ("Inicio", "▲ Início"), ("Rajada", "≋ Rajada"), ("Radar", "📡 Radar")]
MODE_HELP = {
    "Waypoint": "clique no mapa para adicionar waypoints (perto de um fixo, encaixa nele)",
    "Chao": "waypoint no chão: a aeronave mergulha e colide com o terreno",
    "Loiter": "órbita/espera: configure raio, voltas e sentido na aba Rota",
    "Inicio": "define a posição inicial da aeronave atual (rumo = primeiro waypoint)",
    "Radar": "estação de radar: clique no mapa para posicionar (quantas quiser); arraste para mover; configure na aba Radares",
    "Rajada": "rajada de vento localizada: clique no mapa (perto de um waypoint, encaixa nele); arraste a ponta da seta para girar",
}


# parametros do radar na aba Radares: chave -> (rotulo, atributo de RadarStation, divisor de exibicao)
RADAR_FIELDS = {
    "r_range": ("Alcance p/ 1 m² Pd=50% [km]", "range_m", 1000.0),
    "r_height": ("Altura da antena [m]", "height", 1.0),
    "r_scan": ("Período de varredura [s]", "scan_period", 1.0),
    "r_rsig": ("σ distância [m]", "range_sigma", 1.0),
    "r_asig": ("σ azimute [°]", "az_sigma_deg", 1.0),
    "r_esig": ("σ elevação [°]", "el_sigma_deg", 1.0),
    "r_freq": ("Frequência [GHz]", "frequency_ghz", 1.0),
    "r_atm": ("Atenuação atm. [dB/km]", "atm_loss_db_km", 1.0),
    "r_rbias": ("Viés distância [m]", "range_bias", 1.0),
    "r_abias": ("Viés azimute [°]", "az_bias_deg", 1.0),
    "r_emin": ("Elevação mín. [°]", "min_elev_deg", 1.0),
    "r_emax": ("Elevação máx. [°]", "max_elev_deg", 1.0),
    "r_minr": ("Zona cega próxima [m]", "min_range", 1.0),
    "r_azc": ("Setor: centro [°]", "az_center_deg", 1.0),
    "r_azw": ("Setor: abertura [°] (360=total)", "az_width_deg", 1.0),
    "r_fa": ("Falsos alarmes / varredura", "false_alarms", 1.0),
}
RADAR_COLOR = "#ff9f43"


def nice_step(m_per_px: float, px: float = 110.0) -> float:
    raw = m_per_px * px
    for s in (50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000):
        if s >= raw:
            return float(s)
    return 50000.0


class MissionEditor:
    """Logica de edicao (sem GUI): aeronaves, waypoints, fixos, pistas, salvar/carregar."""

    def __init__(self, mission: Mission):
        self.m = mission
        if not self.m.aircraft:
            self.m.aircraft.append(AircraftSpec("uav1"))
        self.cur = 0
        self.mode = "Waypoint"
        self.params = {"agl": 120.0, "speed": 25.0, "radius": 300.0, "turns": 1.0, "dir": 1.0,
                       "g_speed": 10.0, "g_dir": 90.0, "g_radius": 400.0, "g_up": 0.0,
                       **{k: getattr(RadarStation(0, 0), attr) / sc for k, (_l, attr, sc) in RADAR_FIELDS.items()}}

    @property
    def ac(self) -> AircraftSpec:
        return self.m.aircraft[self.cur]

    # ------------------------------------------------------------ aeronaves
    def _unique(self, name: str) -> str:
        while name in [a.name for a in self.m.aircraft]:
            name += "_"
        return name

    def add_aircraft(self, name: str | None = None):
        name = self._unique(name or f"uav{len(self.m.aircraft) + 1}")
        self.m.aircraft.append(AircraftSpec(name, speed=self.params["speed"]))
        self.cur = len(self.m.aircraft) - 1

    def duplicate_aircraft(self):
        src = self.ac
        name = self._unique(f"{src.name}_copia")
        spec = AircraftSpec(name, [Waypoint(**vars(w)) for w in src.waypoints],
                            dict(src.start) if src.start else None, src.speed, src.loop, spline=src.spline, rcs=src.rcs)
        self.m.aircraft.append(spec)
        self.cur = len(self.m.aircraft) - 1

    def remove_aircraft(self):
        if len(self.m.aircraft) > 1:
            self.m.aircraft.pop(self.cur)
            self.cur = max(0, self.cur - 1)

    def rename_aircraft(self, name: str) -> bool:
        name = name.strip()
        if not name or any(a.name == name for k, a in enumerate(self.m.aircraft) if k != self.cur):
            return False
        self.ac.name = name
        return True

    def reverse_route(self):
        """Inverte a ordem dos waypoints (a posicao inicial passa a ser o ultimo ponto, se nao for impacto)."""
        a = self.ac
        a.waypoints.reverse()
        a.start = None

    # -------------------------------------------------------------- objetos
    def snap_waypoint(self, n: float, e: float, r: float) -> tuple[float, float]:
        """Posicao do waypoint (de qualquer aeronave) mais proximo dentro de r metros; senao (n, e)."""
        best = None
        for a in self.m.aircraft:
            for w in a.waypoints:
                d = float(np.hypot(w.n - n, w.e - e))
                if d <= r and (best is None or d < best[0]):
                    best = (d, w.n, w.e)
        return (best[1], best[2]) if best else (n, e)

    def click(self, n: float, e: float, snap: float = 0.0):
        """Aplica um clique (n, e em metros) conforme o modo atual."""
        p = self.params
        if self.mode == "Radar":
            kw = {attr: p[k] * sc for k, (_l, attr, sc) in RADAR_FIELDS.items()}
            k = len(self.m.radars) + 1
            while f"R{k}" in [r.name for r in self.m.radars]:
                k += 1
            self.m.radars.append(RadarStation(n, e, name=f"R{k}", **kw))
            return
        if self.mode == "Rajada":
            n, e = self.snap_waypoint(n, e, snap)
            self.m.gusts.append(GustZone(n, e, p["g_radius"], p["g_speed"], p["g_dir"], p["g_up"]))
            return
        if self.mode == "Inicio":
            hdg = None
            if self.ac.waypoints:
                w = self.ac.waypoints[0]
                hdg = float(np.degrees(np.arctan2(w.e - e, w.n - n)))
            self.ac.start = {"n": n, "e": e, "heading_deg": hdg}
        elif self.mode == "Chao":
            self.ac.waypoints.append(Waypoint(n, e, agl=0.0, speed=p["speed"]))
        elif self.mode == "Loiter":
            self.ac.waypoints.append(Waypoint(n, e, agl=p["agl"], speed=p["speed"], loiter_radius=p["radius"],
                                              loiter_turns=p["turns"], loiter_dir=int(p["dir"])))
        else:
            self.ac.waypoints.append(Waypoint(n, e, agl=p["agl"], speed=p["speed"]))
        if self.ac.waypoints:
            self.ac.speed = self.ac.waypoints[0].speed or p["speed"]     # velocidade de partida

    def hit(self, n: float, e: float, r: float):
        """Objeto mais proximo (dentro de r metros): ('wp'|'start'|'gust'|'gtip'|'radar', i) | None.

        'gtip' e a ponta da seta de direcao da rajada. No modo Rajada, as rajadas tem prioridade sobre waypoints."""
        best = None

        def consider(kind, idx, pn, pe):
            nonlocal best
            d = float(np.hypot(pn - n, pe - e))
            if d <= r and (best is None or d < best[0]):
                best = (d, kind, idx)

        def radars():
            for i, rd in enumerate(self.m.radars):
                consider("radar", i, rd.n, rd.e)

        def gusts():
            for i, g in enumerate(self.m.gusts):
                h = np.radians(g.dir_deg)
                L = 0.7 * g.radius
                consider("gtip", i, g.n + L * np.cos(h), g.e + L * np.sin(h))
                consider("gust", i, g.n, g.e)
        if self.mode == "Radar":
            radars()
        if self.mode == "Rajada":
            gusts()
        for i, w in enumerate(self.ac.waypoints):
            consider("wp", i, w.n, w.e)
        if self.ac.start:
            consider("start", 0, self.ac.start["n"], self.ac.start["e"])
        if self.mode != "Rajada":
            gusts()
        if self.mode != "Radar":
            radars()
        return best[1:] if best else None

    def move(self, kind: str, idx: int, n: float, e: float):
        if kind == "wp":
            w = self.ac.waypoints[idx]
            w.n, w.e = n, e
        elif kind == "start":
            self.ac.start["n"], self.ac.start["e"] = n, e
        elif kind == "radar":
            rd = self.m.radars[idx]
            rd.n, rd.e = n, e
        elif kind == "gust":
            g = self.m.gusts[idx]
            g.n, g.e = self.snap_waypoint(n, e, 0.0)
        elif kind == "gtip":
            g = self.m.gusts[idx]
            g.dir_deg = round(float(np.degrees(np.arctan2(e - g.e, n - g.n))) % 360.0)

    def remove(self, kind: str, idx: int):
        if kind == "wp":
            self.ac.waypoints.pop(idx)
        elif kind == "start":
            self.ac.start = None
        elif kind in ("gust", "gtip"):
            self.m.gusts.pop(idx)
        elif kind == "radar":
            self.m.radars.pop(idx)

    def apply_wp(self, idx: int, agl=None, speed=None, radius=None, turns=None, direction=None):
        """Altera parametros de um waypoint (speed = velocidade do ar do trecho ate ele)."""
        w = self.ac.waypoints[idx]
        if agl is not None:
            w.agl = agl
        if speed is not None:
            w.speed = speed
            if idx == 0:
                self.ac.speed = speed
        if radius is not None:
            w.loiter_radius = radius
        if turns is not None:
            w.loiter_turns = turns
        if direction is not None:
            w.loiter_dir = int(direction)

    def move_wp(self, idx: int, delta: int) -> int:
        w = self.ac.waypoints
        j = idx + delta
        if 0 <= j < len(w):
            w[idx], w[j] = w[j], w[idx]
            return j
        return idx

    def set_all_speed(self, speed: float):
        for w in self.ac.waypoints:
            w.speed = speed
        self.ac.speed = speed

    def undo(self):
        if self.ac.waypoints:
            self.ac.waypoints.pop()

    def clear(self):
        self.ac.waypoints.clear()
        self.ac.start = None

    def save(self, path: str):
        self.m.validate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.m.save(path)

    def load(self, path: str):
        self.m = Mission.load(path)
        if not self.m.aircraft:
            self.m.aircraft.append(AircraftSpec("uav1"))
        self.cur = 0


class BuilderApp:
    def __init__(self, root: tk.Tk, mission: Mission, terrain, path: str, synthetic: bool = False):
        self.root, self.T, self.path, self.synthetic = root, terrain, path, synthetic
        self.ed = MissionEditor(mission)
        self.sel_wp: int | None = None
        self.sel_gust: int | None = None
        self.sel_radar: int | None = None
        self.show_cov = tk.BooleanVar(value=True)
        self._cov_cache = {}
        self._loading = False
        self._drag = None
        self.show_grid = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        root.title("Construtor de missão")
        root.geometry(f"{min(1480, root.winfo_screenwidth() - 40)}x{min(900, root.winfo_screenheight() - 90)}+10+10")
        self._style()
        self._build_ui()
        self._bind_keys()
        self._fitted = False
        self.map.bind("<Configure>", self._first_layout, add="+")
        self._sync_all()
        self.changed()
        self._watch_sat()
        self._use_satellite()

    def _use_satellite(self):
        """Camada padrao: satelite (baixa sob demanda; sem rede/georreferencia volta ao relevo)."""
        self.layer_var.set("Satelite")
        self.map.set_layer("Satelite")

    def _first_layout(self, ev):
        """Ajusta o mapa ao tamanho real do canvas na primeira vez que ele recebe layout."""
        if not self._fitted and ev.width > 200 and ev.height > 200:
            self._fitted = True
            self.map.fit()
            self.map.redraw()

    # ------------------------------------------------------------------ estilo
    def _style(self):
        apply_dark(self.root)

    # ---------------------------------------------------------------------- UI
    def _field(self, parent, label, var, width=9, row=None):
        r = ttk.Frame(parent) if row is None else row
        if row is None:
            r.pack(fill="x", pady=1)
        ttk.Label(r, text=label, width=20).pack(side="left")
        ttk.Entry(r, textvariable=var, width=width).pack(side="left")
        return r

    def _num_var(self, value, cb):
        v = tk.StringVar(value=f"{value:g}")
        v.trace_add("write", lambda *_a: None if self._loading else cb(v))
        return v

    @staticmethod
    def _num(v: tk.StringVar):
        try:
            return float(v.get().replace(",", "."))
        except ValueError:
            return None

    def _build_ui(self):
        root = self.root
        # --- barra de ferramentas
        bar = ttk.Frame(root, style="Bar.TFrame", padding=(8, 6))
        bar.pack(side="top", fill="x")
        self.mode_var = tk.StringVar(value=self.ed.mode)
        for key, label in MODES:
            ttk.Radiobutton(bar, text=label, value=key, variable=self.mode_var, style="Tool.TRadiobutton",
                            command=self._mode_changed).pack(side="left", padx=1)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="↶ Desfazer", command=self._undo).pack(side="left", padx=2)
        ttk.Button(bar, text="Limpar rota", command=self._clear).pack(side="left", padx=2)
        ttk.Button(bar, text="Ajustar vista", command=lambda: (self.map.fit(), self.map.redraw())
                   ).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.layer_var = tk.StringVar(value="Elevacao")
        for name, label in (("Elevacao", "Relevo"), ("Satelite", "Satélite")):
            ttk.Radiobutton(bar, text=label, value=name, variable=self.layer_var, style="Tool.TRadiobutton",
                            command=self._layer_changed).pack(side="left", padx=1)
        for var, text in ((self.show_labels, "Rótulos"), (self.show_grid, "Grade")):
            ttk.Checkbutton(bar, text=text, variable=var, command=lambda: self.map.redraw(),
                            style="Bar.TCheckbutton").pack(side="right", padx=6)
        ttk.Button(bar, text="Salvar (Ctrl+S)", style="Accent.TButton", command=self._save).pack(side="right")

        # --- painel lateral (abas)
        side = ttk.Frame(root, padding=0)
        side.pack(side="right", fill="y")
        nb = ttk.Notebook(side)
        nb.pack(fill="both", expand=True, padx=(0, 2), pady=2)
        self.nb = nb
        tabs = {k: ttk.Frame(nb, padding=8) for k in ("ac", "route", "gust", "radar", "map")}
        nb.add(tabs["ac"], text="Aeronaves")
        nb.add(tabs["route"], text="Rota")
        nb.add(tabs["gust"], text="Rajadas")
        nb.add(tabs["radar"], text="Radares")
        nb.add(tabs["map"], text="Mapa / Arquivo")
        self._tab_aircraft(tabs["ac"])
        self._tab_route(tabs["route"])
        self._tab_gust(tabs["gust"])
        self._tab_radar(tabs["radar"])
        self._tab_map(tabs["map"])

        # --- area do mapa + perfil
        left = ttk.Frame(root, style="Bar.TFrame")
        left.pack(side="left", fill="both", expand=True)
        self.map = MapView(left, self.T, on_click=self._on_click, on_right=self._on_right,
                           on_motion=self._on_motion, draw_overlay=self._overlay, on_press=self._on_press,
                           on_drag=self._on_drag, on_release=self._on_release)
        self.map.pack(fill="both", expand=True)

        self.prof = tk.Canvas(left, height=150, bg=PANEL2, highlightthickness=0)
        self.prof.pack(fill="x")
        self.prof.bind("<Configure>", lambda e: self._profile())
        self.info = ttk.Label(left, text="", anchor="w", style="Bar.TLabel", padding=(8, 2))
        self.info.pack(fill="x")

    def _tab_aircraft(self, f):
        ttk.Label(f, text="AERONAVES", style="Head.TLabel").pack(anchor="w")
        cols = ("nome", "wp")
        self.lb = ttk.Treeview(f, columns=cols, show="headings", height=6, selectmode="browse")
        for c, w, t in (("nome", 150, "Nome"), ("wp", 50, "WP")):
            self.lb.heading(c, text=t)
            self.lb.column(c, width=w, anchor="w" if c == "nome" else "center")
        self.lb.pack(fill="x", pady=4)
        self.lb.bind("<<TreeviewSelect>>", self._on_select)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        for text, cmd in (("+ Nova", self._add_ac), ("⧉ Duplicar", self._dup_ac), ("− Remover", self._rm_ac)):
            ttk.Button(row, text=text, command=cmd).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Label(f, text="PROPRIEDADES DA ATUAL", style="Head.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar()
        r = self._field(f, "Nome / indicativo", self.name_var, 14)
        ttk.Button(r, text="OK", width=4, command=self._rename).pack(side="left", padx=3)
        r.winfo_children()[1].bind("<Return>", lambda e: self._rename())
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Rota em loop (patrulha)", variable=self.loop_var,
                        command=self._loop_changed).pack(anchor="w", pady=4)
        self.spline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Rota suave (spline)", variable=self.spline_var,
                        command=self._spline_changed).pack(anchor="w")
        self.rcs_var = self._num_var(1.0, lambda v: self._rcs_changed(v))
        self._field(f, "RCS radar [m²]", self.rcs_var, 7)
        ttk.Button(f, text="⇄ Inverter rota", command=self._reverse).pack(fill="x", pady=2)

    def _tab_route(self, f):
        ttk.Label(f, text="WAYPOINTS (selecione para editar)", style="Head.TLabel").pack(anchor="w")
        cols = ("i", "agl", "msl", "v", "tipo")
        self.wlb = ttk.Treeview(f, columns=cols, show="headings", height=8, selectmode="browse")
        for c, w, t in (("i", 28, "#"), ("agl", 48, "AGL"), ("msl", 52, "MSL"),
                        ("v", 40, "m/s"), ("tipo", 82, "Tipo")):
            self.wlb.heading(c, text=t)
            self.wlb.column(c, width=w, anchor="center")
        self.wlb.pack(fill="x", pady=4)
        self.wlb.bind("<<TreeviewSelect>>", self._on_wp_select)
        row = ttk.Frame(f)
        row.pack(fill="x")
        for text, cmd in (("▲", lambda: self._shift_wp(-1)), ("▼", lambda: self._shift_wp(1)),
                          ("Remover", self._del_wp)):
            ttk.Button(row, text=text, command=cmd, width=8 if text == "Remover" else 3).pack(
                side="left", expand=text == "Remover", fill="x", padx=1)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Label(f, text="PARÂMETROS (novos pontos / aplicar)", style="Head.TLabel").pack(anchor="w")
        self.vars: dict[str, tk.StringVar] = {}
        for key, label in (("agl", "AGL [m]"), ("speed", "Vel. do trecho [m/s]"),
                           ("radius", "Raio do loiter [m]"), ("turns", "Voltas do loiter")):
            self.vars[key] = self._num_var(self.ed.params[key], lambda v, k=key: self._param_changed(k, v))
            self._field(f, label, self.vars[key])
        r = ttk.Frame(f)
        r.pack(fill="x", pady=1)
        ttk.Label(r, text="Sentido do loiter", width=20).pack(side="left")
        self.dir_var = tk.StringVar(value="Horário")
        cb = ttk.Combobox(r, textvariable=self.dir_var, values=["Horário", "Anti-horário"], state="readonly", width=12)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.ed.params.__setitem__(
            "dir", 1.0 if self.dir_var.get() == "Horário" else -1.0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(8, 2))
        ttk.Button(row, text="Aplicar ao selecionado", command=self._apply_sel).pack(
            side="left", expand=True, fill="x", padx=1)
        ttk.Button(row, text="Vel. p/ todos", command=self._apply_speed_all).pack(
            side="left", expand=True, fill="x", padx=1)

    GUST_KEYS = {"g_speed": "speed", "g_dir": "dir_deg", "g_radius": "radius", "g_up": "up"}

    def _tab_gust(self, f):
        ttk.Label(f, text="RAJADAS NO MAPA (modo ≋, tecla 5)", style="Head.TLabel").pack(anchor="w")
        self.glb = ttk.Treeview(f, columns=("i", "v", "d", "r", "up"), show="headings", height=7, selectmode="browse")
        for c, w, t in (("i", 28, "#"), ("v", 62, "m/s"), ("d", 56, "Rumo°"), ("r", 62, "Raio m"), ("up", 56, "Vert.")):
            self.glb.heading(c, text=t)
            self.glb.column(c, width=w, anchor="center")
        self.glb.pack(fill="x", pady=4)
        self.glb.bind("<<TreeviewSelect>>", self._on_gust_select)
        ttk.Button(f, text="Excluir selecionada", command=self._del_gust).pack(fill="x", pady=2)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Label(f, text="PARÂMETROS (novas / selecionada)", style="Head.TLabel").pack(anchor="w")
        self.gvars = {}
        for key, label in (("g_speed", "Velocidade [m/s]"), ("g_dir", "Sopra para o rumo [°]"),
                           ("g_radius", "Raio de influência [m]"), ("g_up", "Vertical [m/s] (+ sobe)")):
            self.gvars[key] = self._num_var(self.ed.params[key], lambda v, k=key: self._gust_param(k, v))
            self._field(f, label, self.gvars[key])
        ttk.Label(f, text="Clique no mapa com o modo ≋ (perto de um waypoint, encaixa nele).\n"
                          "Arraste o centro para mover e a ponta da seta para girar.\n"
                          "A rajada é máxima no centro e cai suavemente até zero na borda;\n"
                          "a aeronave sente o vento enquanto estiver dentro do círculo.\n"
                          "Rumo: 0 = norte, 90 = leste (para onde o vento sopra).",
                  style="Muted.TLabel", justify="left").pack(anchor="w", pady=8)

    def _gust_param(self, key, var):
        v = self._num(var)
        if v is None or (key == "g_radius" and v <= 0):
            return
        self.ed.params[key] = v
        if self.sel_gust is not None and self.sel_gust < len(self.ed.m.gusts):
            setattr(self.ed.m.gusts[self.sel_gust], self.GUST_KEYS[key], v)
            self.changed()

    def _select_gust(self, i):
        self.sel_gust = i
        if i is not None and i < len(self.ed.m.gusts):
            g = self.ed.m.gusts[i]
            self._loading = True
            for key, attr in self.GUST_KEYS.items():
                self.ed.params[key] = getattr(g, attr)
                self.gvars[key].set(f"{getattr(g, attr):g}")
            self._loading = False
        self._refresh_lists()
        self.map.redraw()

    def _on_gust_select(self, _ev):
        if self._loading:
            return
        sel = self.glb.selection()
        if sel and int(sel[0]) != self.sel_gust:
            self._select_gust(int(sel[0]))

    def _del_gust(self):
        if self.sel_gust is not None and self.sel_gust < len(self.ed.m.gusts):
            self.ed.m.gusts.pop(self.sel_gust)
            self.sel_gust = None
            self.changed()

    def _tab_radar(self, f):
        ttk.Label(f, text="ESTAÇÕES DE RADAR (modo 📡, tecla 6)", style="Head.TLabel").pack(anchor="w")
        self.rlb = ttk.Treeview(f, columns=("nome", "alc", "per", "pos"), show="headings", height=5,
                                selectmode="browse")
        for c, w, t in (("nome", 60, "Nome"), ("alc", 60, "Alc. km"), ("per", 50, "Scan s"), ("pos", 110, "N / E [m]")):
            self.rlb.heading(c, text=t)
            self.rlb.column(c, width=w, anchor="center")
        self.rlb.pack(fill="x", pady=4)
        self.rlb.bind("<<TreeviewSelect>>", self._on_radar_select)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Excluir selecionado", command=self._del_radar).pack(
            side="left", expand=True, fill="x", padx=1)
        ttk.Button(row, text="Duplicar", command=self._dup_radar).pack(side="left", expand=True, fill="x", padx=1)
        self.rname = tk.StringVar()
        r = self._field(f, "Nome", self.rname, 12)
        ttk.Button(r, text="OK", width=4, command=self._rename_radar).pack(side="left", padx=3)
        r.winfo_children()[1].bind("<Return>", lambda e: self._rename_radar())
        ttk.Separator(f).pack(fill="x", pady=6)
        ttk.Label(f, text="PARÂMETROS (novos / selecionado)", style="Head.TLabel").pack(anchor="w")
        self.rvars = {}
        for key, (label, _attr, _sc) in RADAR_FIELDS.items():
            self.rvars[key] = self._num_var(self.ed.params[key], lambda v, k=key: self._radar_param(k, v))
            self._field(f, label, self.rvars[key])
        self.rcov_agl = self._num_var(100.0, lambda v: self._cov_changed())
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(6, 1))
        ttk.Checkbutton(r, text="Mostrar cobertura p/ alvo a", variable=self.show_cov,
                        command=lambda: self.map.redraw()).pack(side="left")
        ttk.Entry(r, textvariable=self.rcov_agl, width=5).pack(side="left", padx=3)
        ttk.Label(r, text="m AGL").pack(side="left")
        self.rcov_rcs = self._num_var(1.0, lambda v: self._cov_changed())
        r = ttk.Frame(f)
        r.pack(fill="x", pady=1)
        ttk.Label(r, text="   e RCS do alvo [m²]").pack(side="left")
        ttk.Entry(r, textvariable=self.rcov_rcs, width=5).pack(side="left", padx=3)
        ttk.Label(f, text="Só se sabe onde as aeronaves estão pelos radares: cada varredura mede\n"
                          "(distância, azimute, elevação) quando o feixe passa pelo alvo; Pd pela equação do\n"
                          "radar (RCS, distância⁴, difração no relevo, curvatura da Terra). Vários radares\n"
                          "são fundidos por IMM-Kalman. Hachurado = Pd ≥ 50% p/ a AGL e RCS indicadas.\n"
                          "Dica: radares em morros enxergam muito mais; use vários para cobrir sombras.",
                  style="Muted.TLabel", justify="left").pack(anchor="w", pady=8)

    def _radar_param(self, key, var):
        v = self._num(var)
        if v is None:
            return
        _l, attr, sc = RADAR_FIELDS[key]
        if (v < 0 and key not in ("r_rbias", "r_abias")) or (key in ("r_range", "r_scan", "r_freq") and v <= 0):
            return
        self.ed.params[key] = v
        if self.sel_radar is not None and self.sel_radar < len(self.ed.m.radars):
            setattr(self.ed.m.radars[self.sel_radar], attr, v * sc)
            self.changed()

    def _select_radar(self, i):
        self.sel_radar = i
        if i is not None and i < len(self.ed.m.radars):
            rd = self.ed.m.radars[i]
            self._loading = True
            self.rname.set(rd.name)
            for key, (_l, attr, sc) in RADAR_FIELDS.items():
                self.ed.params[key] = getattr(rd, attr) / sc
                self.rvars[key].set(f"{getattr(rd, attr) / sc:g}")
            self._loading = False
            self.nb.select(3)
        self._refresh_lists()
        self.map.redraw()

    def _on_radar_select(self, _ev):
        if self._loading:
            return
        sel = self.rlb.selection()
        if sel and int(sel[0]) != self.sel_radar:
            self._select_radar(int(sel[0]))

    def _del_radar(self):
        if self.sel_radar is not None and self.sel_radar < len(self.ed.m.radars):
            self.ed.m.radars.pop(self.sel_radar)
            self.sel_radar = None
            self.changed()

    def _dup_radar(self):
        if self.sel_radar is not None and self.sel_radar < len(self.ed.m.radars):
            src = self.ed.m.radars[self.sel_radar]
            names = {r.name for r in self.ed.m.radars}
            k = len(names) + 1
            while f"R{k}" in names:
                k += 1
            self.ed.m.radars.append(RadarStation(**{**vars(src), "name": f"R{k}", "n": src.n + 500.0}))
            self.sel_radar = len(self.ed.m.radars) - 1
            self.changed()

    def _rename_radar(self):
        name = self.rname.get().strip()
        rs = self.ed.m.radars
        if self.sel_radar is None or self.sel_radar >= len(rs) or not name or \
                any(r.name == name for k, r in enumerate(rs) if k != self.sel_radar):
            self.status.set("nome de radar inválido ou repetido")
            return
        rs[self.sel_radar].name = name
        self.changed()

    def _cov_changed(self):
        if not self._loading:
            self.map.redraw()

    def _tab_map(self, f):
        ttk.Label(f, text="SIMULAÇÃO", style="Head.TLabel").pack(anchor="w")
        m = self.ed.m
        self.wind_n = self._num_var(m.wind_steady_ne[0], lambda v: self._wind_changed())
        self.wind_e = self._num_var(m.wind_steady_ne[1], lambda v: self._wind_changed())
        self.gust = self._num_var(m.gust_sigma, lambda v: self._wind_changed())
        self.dur = self._num_var(m.duration, lambda v: self._dur_changed(v))
        r = self._field(f, "Vento N / E [m/s]", self.wind_n, 5, row=ttk.Frame(f))
        ttk.Entry(r, textvariable=self.wind_e, width=5).pack(side="left", padx=2)
        r.pack(fill="x", pady=1)
        self._field(f, "Rajada σ [m/s]", self.gust)
        self._field(f, "Duração [s]", self.dur)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Label(f, text="LOCAL", style="Head.TLabel").pack(anchor="w")
        self.place_cb = ttk.Combobox(f, values=list(PLACES), state="readonly")
        self.place_cb.pack(fill="x")
        self.place_cb.bind("<<ComboboxSelected>>", self._place_chosen)
        ttk.Label(f, text="Centro (lat, lon) e lado [km]", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))
        o = self.ed.m.origin
        self.geo = {k: tk.StringVar(value=f"{o[k]:g}") for k in ("lat", "lon", "size_km")}
        r = ttk.Frame(f)
        r.pack(fill="x")
        for k in ("lat", "lon", "size_km"):
            ttk.Entry(r, textvariable=self.geo[k], width=9).pack(side="left", padx=1)
        ttk.Button(f, text="Recentrar mapa", command=self._recenter).pack(fill="x", pady=3)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Label(f, text="ARQUIVO", style="Head.TLabel").pack(anchor="w")
        self.path_var = tk.StringVar(value=self.path)
        ttk.Entry(f, textvariable=self.path_var, width=36).pack(fill="x", pady=2)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Abrir...", command=self._open).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(row, text="Salvar como...", command=self._save_as).pack(side="left", expand=True, fill="x", padx=1)
        self.status = tk.StringVar(value=MODE_HELP["Waypoint"])
        ttk.Label(f, textvariable=self.status, wraplength=300, justify="left", style="Muted.TLabel").pack(
            anchor="w", pady=10)

    def _bind_keys(self):
        r = self.root
        r.bind("<Control-s>", lambda e: self._save())
        r.bind("<Control-z>", lambda e: self._undo())
        r.bind("<Escape>", lambda e: self._select_wp(None))
        r.bind("<Delete>", lambda e: self._del_wp() if self._not_typing() else None)
        for i, (key, _l) in enumerate(MODES):
            r.bind(str(i + 1), lambda e, k=key: self._set_mode(k) if self._not_typing() else None)

    def _not_typing(self) -> bool:
        return self.root.focus_get().winfo_class() not in ("TEntry", "Entry", "TCombobox")

    # --------------------------------------------------------- sincronizacao
    def _sync_all(self):
        """Preenche campos a partir do modelo (apos abrir arquivo / trocar de aeronave)."""
        self._loading = True
        m, a = self.ed.m, self.ed.ac
        self.name_var.set(a.name)
        self.loop_var.set(a.loop)
        self.spline_var.set(a.spline)
        self.rcs_var.set(f"{a.rcs:g}")
        self.wind_n.set(f"{m.wind_steady_ne[0]:g}")
        self.wind_e.set(f"{m.wind_steady_ne[1]:g}")
        self.gust.set(f"{m.gust_sigma:g}")
        self.dur.set(f"{m.duration:g}")
        self.vars["speed"].set(f"{a.speed:g}")
        self.ed.params["speed"] = a.speed
        for k in self.geo:
            self.geo[k].set(f"{m.origin[k]:g}")
        self.sel_gust = None
        self.sel_radar = None
        self._loading = False

    def changed(self, lists: bool = True):
        """Redesenha tudo (e atualiza as listas laterais)."""
        if lists:
            self._refresh_lists()
        self.map.redraw()
        self._profile()

    def _refresh_lists(self):
        self._loading = True
        ed = self.ed
        self.lb.delete(*self.lb.get_children())
        for k, a in enumerate(ed.m.aircraft):
            self.lb.tag_configure(f"c{k}", foreground=COLORS[k % len(COLORS)])
            self.lb.insert("", "end", iid=str(k), values=(a.name, len(a.waypoints)),
                           tags=(f"c{k}",))
        self.lb.selection_set(str(ed.cur))
        self.wlb.delete(*self.wlb.get_children())
        for i, w in enumerate(ed.ac.waypoints):
            if w.impact:
                tipo, agl, msl = "CHÃO", "0", "-"
            else:
                tipo = f"órbita r{w.loiter_radius:.0f}×{w.loiter_turns:g}" if w.loiter_radius > 0 else "voo"
                agl = f"{w.agl:.0f}"
                msl = f"{float(self.T.height(w.n, w.e)) + w.agl:.0f}" if self.T.inside(w.n, w.e) else "-"
            self.wlb.insert("", "end", iid=str(i), values=(i + 1, agl, msl,
                                                           f"{(w.speed or ed.ac.speed):.0f}", tipo))
        if self.sel_wp is not None and self.sel_wp < len(ed.ac.waypoints):
            self.wlb.selection_set(str(self.sel_wp))
        self.glb.delete(*self.glb.get_children())
        for i, g in enumerate(ed.m.gusts):
            self.glb.insert("", "end", iid=str(i), values=(i + 1, f"{g.speed:g}", f"{g.dir_deg:03.0f}",
                                                           f"{g.radius:g}", f"{g.up:+g}"))
        if self.sel_gust is not None and self.sel_gust < len(ed.m.gusts):
            self.glb.selection_set(str(self.sel_gust))
        self.rlb.delete(*self.rlb.get_children())
        for i, rd in enumerate(ed.m.radars):
            self.rlb.insert("", "end", iid=str(i), values=(rd.name, f"{rd.range_m / 1000:g}", f"{rd.scan_period:g}",
                                                           f"{rd.n:.0f} / {rd.e:.0f}"))
        if self.sel_radar is not None and self.sel_radar < len(ed.m.radars):
            self.rlb.selection_set(str(self.sel_radar))
        self._loading = False

    # ------------------------------------------------------------ eventos (mapa)
    def _on_press(self, n, e):
        h = self.ed.hit(n, e, 14 / self.map.scale)
        self._drag = h
        return h is not None

    def _on_drag(self, n, e):
        if not self._drag:
            return
        kind, idx = self._drag
        self.ed.move(kind, idx, n, e)
        if kind == "gtip":
            self._loading = True
            self.gvars["g_dir"].set(f"{self.ed.m.gusts[idx].dir_deg:g}")
            self._loading = False
        self.map.redraw()
        self._profile()

    def _on_release(self, moved):
        h, self._drag = self._drag, None
        if moved:
            self.changed()
            return
        if not h:
            return
        if h[0] == "wp":
            self._select_wp(h[1])
        elif h[0] in ("gust", "gtip"):
            self._select_gust(h[1])
        elif h[0] == "radar":
            self._select_radar(h[1])

    def _on_click(self, n, e):
        self.sel_wp = None
        self.ed.click(n, e, snap=14 / self.map.scale)
        if self.ed.mode == "Inicio":
            self.status.set("início definido")
        elif self.ed.mode == "Rajada":
            self.sel_gust = len(self.ed.m.gusts) - 1
        elif self.ed.mode == "Radar":
            self._select_radar(len(self.ed.m.radars) - 1)
        self.changed()

    def _on_right(self, n, e):
        h = self.ed.hit(n, e, 25 / self.map.scale)
        if h:
            self.ed.remove(*h)
            self.sel_wp = self.sel_gust = self.sel_radar = None
            self.changed()

    def _on_motion(self, n, e):
        if self.T.inside(n, e):
            lat, lon = self.T.to_latlon(n, e)
            self.info.config(text=f"N={n:8.0f} m  E={e:8.0f} m  elev={float(self.T.height(n, e)):5.0f} m"
                                  f"   lat={float(lat):.5f}  lon={float(lon):.5f}")

    # ---------------------------------------------------------- eventos (UI)
    def _set_mode(self, key):
        self.mode_var.set(key)
        self._mode_changed()

    def _mode_changed(self):
        self.ed.mode = self.mode_var.get()
        self.status.set(MODE_HELP[self.ed.mode])

    def _layer_changed(self):
        self.map.set_layer(self.layer_var.get())

    def _undo(self):
        self.ed.undo()
        self.sel_wp = None
        self.changed()

    def _clear(self):
        self.ed.clear()
        self.sel_wp = None
        self.changed()

    def _select_wp(self, k):
        self.sel_wp = k
        if k is None:
            self.sel_gust = self.sel_radar = None
        if k is not None and k < len(self.ed.ac.waypoints):
            w = self.ed.ac.waypoints[k]
            self._loading = True
            self.vars["agl"].set(f"{max(w.agl, 0):g}")
            self.vars["speed"].set(f"{(w.speed or self.ed.ac.speed):g}")
            if w.loiter_radius > 0:
                self.vars["radius"].set(f"{w.loiter_radius:g}")
                self.vars["turns"].set(f"{w.loiter_turns:g}")
                self.dir_var.set("Horário" if w.loiter_dir >= 0 else "Anti-horário")
            for key in ("agl", "speed", "radius", "turns"):
                self._param_changed(key, self.vars[key], force=True)
            self._loading = False
            self.status.set(f"waypoint {k + 1} selecionado: edite na aba Rota e clique 'Aplicar ao selecionado'")
        self.map.redraw()
        self._loading = True
        self.wlb.selection_remove(*self.wlb.selection())
        if k is not None and str(k) in self.wlb.get_children():
            self.wlb.selection_set(str(k))
        self._loading = False

    def _on_wp_select(self, _ev):
        if self._loading:
            return
        sel = self.wlb.selection()
        if sel and int(sel[0]) != self.sel_wp:
            self._select_wp(int(sel[0]))

    def _shift_wp(self, d):
        if self.sel_wp is None:
            return
        self.sel_wp = self.ed.move_wp(self.sel_wp, d)
        self.changed()

    def _del_wp(self):
        if self.sel_wp is None:
            self._del_radar() if self.sel_radar is not None else self._del_gust()
        elif self.sel_wp < len(self.ed.ac.waypoints):
            self.ed.ac.waypoints.pop(self.sel_wp)
            self.sel_wp = None
            self.changed()

    def _apply_sel(self):
        if self.sel_wp is None or self.sel_wp >= len(self.ed.ac.waypoints):
            self.status.set("selecione um waypoint (na lista ou clicando nele no mapa)")
            return
        p = self.ed.params
        w = self.ed.ac.waypoints[self.sel_wp]
        loit = w.loiter_radius > 0
        self.ed.apply_wp(self.sel_wp, agl=(None if w.impact else p["agl"]), speed=p["speed"],
                         radius=(p["radius"] if loit else None), turns=(p["turns"] if loit else None),
                         direction=(p["dir"] if loit else None))
        self.status.set(f"waypoint {self.sel_wp + 1}: {p['speed']:g} m/s no trecho até ele")
        self.changed()

    def _apply_speed_all(self):
        self.ed.set_all_speed(self.ed.params["speed"])
        self.status.set(f"{self.ed.params['speed']:g} m/s em todos os trechos")
        self.changed()

    def _param_changed(self, key, var, force=False):
        if self._loading and not force:
            return
        v = self._num(var)
        if v is None:
            return
        self.ed.params[key] = v
        if key == "speed" and not force:
            self.ed.ac.speed = v
            self.changed(lists=False)

    def _on_select(self, _ev):
        if self._loading:
            return
        sel = self.lb.selection()
        if sel and int(sel[0]) != self.ed.cur:          # o evento tambem dispara em selection_set programatico
            self.ed.cur = int(sel[0])
            self.sel_wp = None
            self._sync_all()
            self.changed()

    def _add_ac(self):
        self.ed.add_aircraft()
        self._sync_all()
        self.changed()

    def _dup_ac(self):
        self.ed.duplicate_aircraft()
        self._sync_all()
        self.changed()

    def _rm_ac(self):
        self.ed.remove_aircraft()
        self._sync_all()
        self.changed()

    def _rename(self):
        if not self.ed.rename_aircraft(self.name_var.get()):
            self.status.set("nome vazio ou já usado por outra aeronave")
            self.name_var.set(self.ed.ac.name)
            return
        self.changed()

    def _rcs_changed(self, var):
        v = self._num(var)
        if v is not None and v > 0:
            self.ed.ac.rcs = v

    def _loop_changed(self):
        self.ed.ac.loop = self.loop_var.get()
        self.changed()

    def _spline_changed(self):
        self.ed.ac.spline = self.spline_var.get()
        self.changed()

    def _reverse(self):
        self.ed.reverse_route()
        self.sel_wp = None
        self.changed()

    def _wind_changed(self):
        n, e, g = self._num(self.wind_n), self._num(self.wind_e), self._num(self.gust)
        if None not in (n, e, g):
            self.ed.m.wind_steady_ne, self.ed.m.gust_sigma = (n, e), max(g, 0.0)

    def _dur_changed(self, v):
        x = self._num(v)
        if x is not None and x >= 10:
            self.ed.m.duration = x
            self.changed(lists=False)

    # ------------------------------------------------------------ linha do tempo

    # --------------------------------------------------------------- mapa / arquivo
    def _watch_sat(self):
        m = self.map
        if m.sat_state == "baixando":
            self.status.set(m.sat_msg)
        elif m.sat_state == "erro":
            self.status.set(m.sat_msg)
            self.layer_var.set("Elevacao")
            m.sat_state = "nao"
        elif m.sat_state == "pronto" and m.sat_msg:
            self.status.set(m.sat_msg)
            m.sat_msg = ""
        self.root.after(300, self._watch_sat)

    def _place_chosen(self, _ev):
        lat, lon, size = PLACES[self.place_cb.get()]
        for k, v in zip(("lat", "lon", "size_km"), (lat, lon, size)):
            self.geo[k].set(f"{v:g}")
        self._recenter()

    def _recenter(self):
        try:
            lat, lon, size = (float(self.geo[k].get().replace(",", ".")) for k in ("lat", "lon", "size_km"))
        except ValueError:
            self.status.set("lat/lon/lado inválidos")
            return
        m = self.ed.m
        if any(a.waypoints or a.start for a in m.aircraft) or m.gusts or m.radars:
            if not messagebox.askyesno("Recentrar", "Recentrar o mapa apaga waypoints, rajadas e radares (as coordenadas "
                                                    "são relativas ao centro). Continuar?"):
                return
        try:
            T = load_terrain({"lat": lat, "lon": lon, "size_km": size}, self.synthetic)
        except Exception as ex:
            self.status.set(f"falha ao carregar DEM: {ex}")
            return
        self.T = T
        m.origin = {"lat": lat, "lon": lon, "size_km": size}
        for a in m.aircraft:
            a.waypoints.clear()
            a.start = None
        m.gusts.clear()
        m.radars.clear()
        self.sel_wp = self.sel_radar = None
        self._cov_cache.clear()
        self.map.set_terrain(T)
        self._use_satellite()
        self.map.fit()
        self.changed()
        self.status.set(f"mapa recentrado em {lat:.4f}, {lon:.4f} ({size:g} km)")

    def _save(self, path: str | None = None):
        path = path or self.path_var.get()
        try:
            self.ed.save(path)
        except ValueError as ex:
            self.status.set(f"NÃO SALVO: {ex}")
            messagebox.showwarning("Missão inválida", str(ex))
            return
        self.path = path
        self.path_var.set(path)
        self.status.set(f"salvo em {path}\nsimule: python run_mission.py {path}")

    def _save_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", initialdir="missions",
                                         filetypes=[("Missão", "*.json")])
        if p:
            self._save(p)

    def _open(self):
        p = filedialog.askopenfilename(initialdir="missions", filetypes=[("Missão", "*.json")])
        if not p:
            return
        self.ed.load(p)
        self.path = p
        self.path_var.set(p)
        try:
            self.T = load_terrain(self.ed.m.origin, self.synthetic)
            self.map.set_terrain(self.T)
            self._use_satellite()
        except Exception as ex:
            self.status.set(f"falha ao carregar DEM: {ex}")
            return
        self.sel_wp = None
        self._sync_all()
        self.changed()
        self.status.set(f"carregado: {p}")

    # -------------------------------------------------------------------- desenho
    def _overlay(self, c: MapView, light: bool):
        if self.show_grid.get():
            self._draw_grid(c)
        for i, rd in enumerate(self.ed.m.radars):
            self._draw_radar(c, i, rd)
        for i, g in enumerate(self.ed.m.gusts):
            self._draw_gust(c, i, g)
        for k, a in enumerate(self.ed.m.aircraft):
            self._draw_route(c, k, a)

    def _coverage(self, rd: RadarStation, agl: float, rcs: float = 1.0):
        key = (rd.n, rd.e, rd.height, rd.range_m, rd.min_range, rd.min_elev_deg, rd.max_elev_deg,
               rd.az_center_deg, rd.az_width_deg, rd.frequency_ghz, rd.atm_loss_db_km, rd.pfa, agl, rcs, id(self.T))
        if key not in self._cov_cache:
            if len(self._cov_cache) > 24:
                self._cov_cache.clear()
            ns, es, ok = coverage_mask(self.T, rd, agl, rcs)
            runs = []                         # corridas horizontais (poucos itens no canvas)
            hn, he = (ns[1] - ns[0]) / 2, (es[1] - es[0]) / 2
            for i in range(ok.shape[0]):
                j = 0
                while j < ok.shape[1]:
                    if ok[i, j]:
                        k = j
                        while k + 1 < ok.shape[1] and ok[i, k + 1]:
                            k += 1
                        runs.append((ns[i] - hn, ns[i] + hn, es[j] - he, es[k] + he))
                        j = k + 1
                    else:
                        j += 1
            self._cov_cache[key] = runs
        return self._cov_cache[key]

    def _draw_radar(self, c: MapView, i: int, rd: RadarStation):
        sel = i == self.sel_radar
        col = RADAR_COLOR
        if self.show_cov.get() and (sel or len(self.ed.m.radars) <= 6):
            for n0, n1, e0, e1 in self._coverage(rd, self._num(self.rcov_agl) or 100.0, self._num(self.rcov_rcs) or 1.0):
                x0, y0 = c.w2s(n1, e0)
                x1, y1 = c.w2s(n0, e1)
                c.create_rectangle(x0, y0, x1 + 1, y1 + 1, fill=col, outline="",
                                   stipple="gray25" if sel else "gray12")
        x, y = c.w2s(rd.n, rd.e)
        R = rd.range_m * c.scale
        if rd.az_width_deg >= 360.0:
            c.create_oval(x - R, y - R, x + R, y + R, outline=col, width=2 if sel else 1, dash=(8, 5))
        else:                                 # setor: arco + dois raios
            a0 = 90.0 - (rd.az_center_deg + rd.az_width_deg / 2)
            c.create_arc(x - R, y - R, x + R, y + R, start=a0, extent=rd.az_width_deg, style="arc",
                         outline=col, width=2 if sel else 1)
            for ang in (rd.az_center_deg - rd.az_width_deg / 2, rd.az_center_deg + rd.az_width_deg / 2):
                h = np.radians(ang)
                c.create_line(x, y, x + R * np.sin(h), y - R * np.cos(h), fill=col, dash=(4, 4))
        c.create_polygon(x, y - 11, x + 10, y + 7, x - 10, y + 7, fill=col, outline="black", width=2)
        c.create_oval(x - 3, y - 2, x + 3, y + 4, fill="black", outline="")
        if sel:
            c.create_oval(x - 16, y - 16, x + 16, y + 16, outline="yellow", width=3)
        txt = f"{rd.name}  {rd.range_m / 1000:g} km  {rd.scan_period:g}s"
        c.create_text(x + 13, y - 10, text=txt, fill="white", anchor="sw", font=("Segoe UI", 9, "bold"))

    def _draw_gust(self, c: MapView, i: int, g: GustZone):
        sel = i == self.sel_gust
        col = "#5ad1ff"
        x, y = c.w2s(g.n, g.e)
        R = g.radius * c.scale
        c.create_oval(x - R, y - R, x + R, y + R, outline=col, width=3 if sel else 2, dash=(7, 4))
        c.create_oval(x - R / 2, y - R / 2, x + R / 2, y + R / 2, outline=col, width=1, dash=(2, 4))
        h = np.radians(g.dir_deg)
        L = max(0.7 * R, 22.0)
        tx, ty = x + L * np.sin(h), y - L * np.cos(h)
        c.create_line(x, y, tx, ty, fill="black", width=6, arrow="last", arrowshape=(14, 16, 6))
        c.create_line(x, y, tx, ty, fill=col, width=3, arrow="last", arrowshape=(12, 14, 5))
        c.create_polygon(x, y - 6, x + 6, y, x, y + 6, x - 6, y, fill=col, outline="black", width=2)
        if sel:
            c.create_oval(tx - 7, ty - 7, tx + 7, ty + 7, outline="yellow", width=2)
        txt = f"rajada {i + 1}: {g.speed:g} m/s → {g.dir_deg:03.0f}°" + (f"  ↕{g.up:+g}" if g.up else "")
        c.create_text(x, y - max(R, 14) - 6, text=txt, fill="white", anchor="s", font=("Segoe UI", 9, "bold"))

    def _draw_grid(self, c: MapView):
        W, H = c.winfo_width(), c.winfo_height()
        step = nice_step(1.0 / c.scale)
        n_top, e_left = c.s2w(0, 0)
        n_bot, e_right = c.s2w(W, H)
        e0 = np.ceil(e_left / step) * step
        while e0 < e_right:
            x, _ = c.w2s(0, e0)
            c.create_line(x, 0, x, H, fill="#8fa3bf", dash=(2, 8))
            c.create_text(x + 3, H - 3, text=f"E {e0 / 1000:g} km", fill="#b7c4d8", anchor="sw", font=("Segoe UI", 8))
            e0 += step
        n0 = np.ceil(n_bot / step) * step
        while n0 < n_top:
            _, y = c.w2s(n0, 0)
            c.create_line(0, y, W, y, fill="#8fa3bf", dash=(2, 8))
            c.create_text(3, y - 2, text=f"N {n0 / 1000:g} km", fill="#b7c4d8", anchor="sw", font=("Segoe UI", 8))
            n0 += step
        # barra de escala + norte
        bar = step * c.scale
        x0, y0 = W - 24 - bar, H - 26
        c.create_rectangle(x0 - 8, y0 - 22, W - 10, y0 + 14, fill="#0b0e13", outline=LINE)
        c.create_line(x0, y0, x0 + bar, y0, fill="white", width=3)
        c.create_text(x0 + bar / 2, y0 - 9, text=f"{step / 1000:g} km" if step >= 1000 else f"{step:g} m",
                      fill="white", font=("Segoe UI", 9, "bold"))
        c.create_text(W - 30, 24, text="N", fill="white", font=("Segoe UI", 11, "bold"))
        c.create_line(W - 30, 56, W - 30, 36, fill="white", width=3, arrow="last")

    def _draw_route(self, c: MapView, k: int, a: AircraftSpec):
        col = COLORS[k % len(COLORS)]
        cur = k == self.ed.cur
        wd = 3 if cur else 1.5
        pts = []
        if a.start:
            pts.append((a.start["n"], a.start["e"]))
        pts += [(w.n, w.e) for w in a.waypoints]
        if a.loop and len(a.waypoints) > 2:
            pts.append((a.waypoints[0].n, a.waypoints[0].e))
        if len(pts) >= 2:
            if a.spline:                                      # curva suave + poligono de controle tracejado
                c.create_line(*[v for p in pts for v in c.w2s(*p)], fill=col, width=1, dash=(2, 4))
                dense = a.route_waypoints()
                dp = ([(a.start["n"], a.start["e"])] if a.start else []) + [(w.n, w.e) for w in dense]
                if a.loop and len(a.waypoints) > 2:
                    dp.append((a.waypoints[0].n, a.waypoints[0].e))
                if len(dp) >= 2:
                    c.create_line(*[v for p in dp for v in c.w2s(*p)], fill=col, width=wd + 1, smooth=False)
            else:
                c.create_line(*[v for p in pts for v in c.w2s(*p)], fill=col, width=wd)
            for p, q in zip(pts[:-1], pts[1:]):                # setas de sentido no meio de cada trecho
                (x1, y1), (x2, y2) = c.w2s(*p), c.w2s(*q)
                if np.hypot(x2 - x1, y2 - y1) < 40:
                    continue
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ang = np.arctan2(y2 - y1, x2 - x1)
                dx, dy = np.cos(ang) * 6, np.sin(ang) * 6
                c.create_polygon(mx + dx, my + dy, mx - dx - dy * .7, my - dy + dx * .7,
                                 mx - dx + dy * .7, my - dy - dx * .7, fill=col, outline="black")
                if cur and self.show_labels.get() and np.hypot(x2 - x1, y2 - y1) > 90:
                    d = np.hypot(q[0] - p[0], q[1] - p[1])
                    brg = (np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))) % 360
                    c.create_text(mx - dy * 2.2, my + dx * 2.2, text=f"{brg:03.0f}°  {d / 1000:.2f} km",
                                  fill="white", font=("Segoe UI", 8))
        if a.start:
            x, y = c.w2s(a.start["n"], a.start["e"])
            c.create_polygon(x, y - 10, x - 8, y + 6, x + 8, y + 6, fill=col, outline="black", width=2)
            c.create_text(x + 12, y, text=f"{a.name} (início)",
                          fill=col, anchor="w", font=("Segoe UI", 9, "bold"))
        for i, w in enumerate(a.waypoints):
            x, y = c.w2s(w.n, w.e)
            if w.impact:
                c.create_line(x - 9, y - 9, x + 9, y + 9, fill="black", width=5)
                c.create_line(x - 9, y + 9, x + 9, y - 9, fill="black", width=5)
                c.create_line(x - 8, y - 8, x + 8, y + 8, fill=col, width=2)
                c.create_line(x - 8, y + 8, x + 8, y - 8, fill=col, width=2)
            else:
                c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=col, outline="black", width=2)
            if w.loiter_radius > 0:
                rr = w.loiter_radius * c.scale
                c.create_oval(x - rr, y - rr, x + rr, y + rr, outline=col, dash=(5, 3), width=2)
            if cur and i == self.sel_wp:
                c.create_oval(x - 12, y - 12, x + 12, y + 12, outline="yellow", width=3)
            v = w.speed or a.speed
            txt = f"{i + 1}" if not cur else f"{a.name}:{i + 1}  {v:.0f} m/s  {max(w.agl, 0):.0f} m AGL"
            c.create_text(x + 9, y - 9, text=txt, fill="white", anchor="sw", font=("Segoe UI", 9, "bold"))

    def _profile(self):
        pc = self.prof
        pc.delete("all")
        W, H = max(pc.winfo_width(), 100), max(pc.winfo_height(), 60)
        ac = self.ed.ac
        pts = ([(ac.start["n"], ac.start["e"])] if ac.start else []) + [(w.n, w.e) for w in ac.waypoints]
        if len(pts) < 2:
            pc.create_text(10, 10, anchor="nw", fill=MUTED,
                           text=f"{ac.name}: adicione ≥ 2 pontos para ver o perfil do terreno")
            return
        P = np.array(pts)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        s = np.linspace(0, cum[-1], 300)
        h = self.T.height(np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1]))
        nodes = [max(w.agl, 0.0) for w in ac.waypoints]
        if ac.start:
            nodes = [nodes[0] if nodes and nodes[0] > 0 else 100.0] + nodes
        plan = h + np.interp(s, cum, nodes)
        lo, hi = h.min() - 20, max(plan.max(), h.max()) + 30
        X = lambda v: 45 + v / max(cum[-1], 1) * (W - 60)
        Y = lambda v: H - 18 - (v - lo) / (hi - lo) * (H - 40)
        poly = [(45, H - 18)] + [(X(a), Y(b)) for a, b in zip(s, h)] + [(X(cum[-1]), H - 18)]
        pc.create_polygon([v for p in poly for v in p], fill="#7a5a43", outline="")
        pc.create_line([v for a, b in zip(s, plan) for v in (X(a), Y(b))],
                       fill=COLORS[self.ed.cur % len(COLORS)], width=2)
        pc.create_text(4, Y(hi), anchor="nw", text=f"{hi:.0f} m", fill=MUTED)
        pc.create_text(4, Y(lo), anchor="sw", text=f"{lo:.0f} m", fill=MUTED)
        slope = np.degrees(np.arctan(np.abs(np.gradient(h, s)))).max()
        v_leg = np.array(([ac.speed] if ac.start else []) + [(w.speed or ac.speed) for w in ac.waypoints])
        t_est = float(np.sum(np.diff(cum) / np.maximum(v_leg[1:], 1.0)))
        pc.create_text(W - 8, 4, anchor="ne", fill=FG, text=(
            f"{ac.name}: {cum[-1] / 1000:.1f} km, ~{t_est / 60:.1f} min (velocidade por trecho), "
            f"declividade máx. {slope:.0f}°"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=os.path.join("missions", "missao.json"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--size", type=float, help="lado do mapa [km]")
    ap.add_argument("--place", help="local pronto, ex.: itajuba, \"pouso alegre\", taubate")
    args = ap.parse_args()
    if os.path.exists(args.file):
        m = Mission.load(args.file)
    else:
        m = Mission(name=os.path.splitext(os.path.basename(args.file))[0])
        if args.place:
            m.origin = place_origin(args.place)
        for k, v in (("lat", args.lat), ("lon", args.lon), ("size_km", args.size)):
            if v is not None:
                m.origin[k] = v
    terrain = load_terrain(m.origin, args.synthetic)
    root = tk.Tk()
    BuilderApp(root, m, terrain, args.file, args.synthetic)
    root.mainloop()


if __name__ == "__main__":
    main()
