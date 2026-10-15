"""Executa uma missao salva pelo construtor (mission_builder.py).

Uso:
    python scripts/missoes/run_mission.py missions/minha.json                # simula ate todas as aeronaves cairem/sairem do mapa
                                                             # (limite: duracao da missao) e abre o replay
    python scripts/missoes/run_mission.py missions/minha.json --duration 300 # limita a 5 min
    python scripts/missoes/run_mission.py missions/minha.json --batch        # sem janela: mesma simulacao e salva PNGs
    python scripts/missoes/run_mission.py missions/minha.json --resim        # ignora o cache e simula de novo

A simulacao e guardada em results/cache/ (chave = missao + duracao + DEM + codigo de swarm_sim/); abrir o
replay de uma missao ja simulada e instantaneo. Editou a missao ou o codigo? O cache se invalida sozinho.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from swarm_sim.mission import Mission, load_terrain                 # noqa: E402


def cache_path(m: Mission, dur: float, synthetic: bool, out: str) -> str:
    """Caminho do cache: hash da missao, duracao, DEM sintetico/real e do codigo da simulacao."""
    h = hashlib.sha1(json.dumps([m.to_dict(), dur, synthetic], sort_keys=True).encode())
    for f in sorted(glob.glob(os.path.join(ROOT, "swarm_sim", "*.py"))):
        if os.path.basename(f) not in ("viewer.py", "viz.py", "mapview.py", "theme.py", "tiles.py"):
            with open(f, "rb") as fh:
                h.update(fh.read())
    return os.path.join(out, "cache", f"{m.name}_{h.hexdigest()[:12]}.pkl")


def load_cached(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:                      # inexistente, corrompido ou de outra versao: simula de novo
        return None


def save_cache(sim, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for old in glob.glob(os.path.join(os.path.dirname(path), os.path.basename(path).rsplit("_", 1)[0] + "_*.pkl")):
        os.remove(old)                     # um cache por missao
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def report(sim):
    for row in sim.summary():
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
    for ev in sim.events:
        print("EVENTO", ev)
    if len(sim.agents) > 1:
        print(f"menor distancia entre aeronaves: {sim.min_pair_dist:.1f} m")
    if sim.radar_net:
        names = [r.name for r in sim.radar_net.radars]
        for row in sim.radar_net.summary([a.name for a in sim.agents], sim.t):
            det = ", ".join(f"{n}={c}" for n, c in zip(names, row["deteccoes"]))
            if row["rmse_pos"] is None:
                print(f"RADAR {row['agent']}: nunca rastreada (deteccoes: {det})")
            else:
                print(f"RADAR {row['agent']}: rastreada {100 * row['rastreada']:.0f}% do tempo, 1a pista em "
                      f"{row['t_1a_pista']:.0f}s, erro RMS {row['rmse_pos']:.0f} m (deteccoes: {det})")
        fa = sum(1 for d in sim.radar_net.detections if d.truth is None)
        print(f"RADAR: {len(sim.radar_net.detections)} deteccoes ({fa} falsos alarmes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission", help="arquivo .json da missao")
    ap.add_argument("--duration", type=float,
                    help="limite de duracao simulada [s] (padrao: a da missao; a simulacao termina antes se todas "
                         "as aeronaves colidirem ou sairem do mapa)")
    ap.add_argument("--batch", action="store_true", help="sem janela; salva graficos em --out")
    ap.add_argument("--synthetic", action="store_true", help="usa DEM fractal em vez do GeoTIFF")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--resim", action="store_true", help="ignora o cache e simula de novo")
    ap.add_argument("--seed", type=int, default=0,
                    help="semente do sorteio quando a missao e um preset aleatorio (raio de ponto, faixa de vel./AGL)")
    args = ap.parse_args()

    m = Mission.load(args.mission)
    if m.is_preset:
        m = m.randomized(args.seed)
        print(f"preset aleatorio: instancia sorteada com --seed {args.seed}")
    dur = args.duration or m.duration
    cpath = cache_path(m, dur, args.synthetic, args.out)
    sim = None if args.resim else load_cached(cpath)
    if sim is not None:
        print(f"missao '{m.name}': simulacao em cache ({sim.t:.0f}s), sem reprocessar (--resim forca)")
    else:
        terrain = load_terrain(m.origin, args.synthetic)
        sim = m.build(terrain)
        print(f"missao '{m.name}': {len(m.aircraft)} aeronave(s), ate {dur:.0f}s (ou ate todas cairem/sairem do mapa)")
        t0 = time.time()
        sim.run(dur, progress=args.batch)
        print(f"simulado {sim.t:.0f}s em {time.time() - t0:.1f}s de relogio")
        try:
            save_cache(sim, cpath)
        except Exception as ex:             # cache e so otimizacao
            print("aviso: nao foi possivel gravar o cache:", ex)
    report(sim)

    if args.batch:
        import matplotlib
        matplotlib.use("Agg")
        from swarm_sim.viz import plot_3d, plot_map, plot_profiles
        os.makedirs(args.out, exist_ok=True)
        tag = os.path.splitext(os.path.basename(args.mission))[0]
        plot_map(sim, os.path.join(args.out, f"{tag}_mapa.png"))
        plot_profiles(sim, os.path.join(args.out, f"{tag}_perfis.png"))
        plot_3d(sim, os.path.join(args.out, f"{tag}_3d.png"))
        print("graficos em", args.out)
    else:
        from swarm_sim.viewer import play
        play(sim, f"Replay - {m.name}")


if __name__ == "__main__":
    main()
