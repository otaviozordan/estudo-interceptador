"""Replay de um episodio de reabastecimento (ambiente cinematico de swarm_sim/refuel).

    python view_refuel.py missions/.../missao.json                         # politica hungaro_prioridade
    python view_refuel.py missions/.../missao.json --policy results/refuel/<carimbo>/il_ppo.pt --seed 3
    python view_refuel.py missions/.../missao.json --policy guloso_proximo --save ep.gif

--policy: nome de uma linha de base (aleatorio, guloso_proximo, guloso_urgente, guloso_prioridade, hungaro,
hungaro_prioridade, hungaro_prio_sem_filtro) ou pesos .pt salvos por run_refuel.py (il.pt, ppo.pt, il_ppo.pt).
Legenda do mapa: ver swarm_sim/refuel/replay.py. Teclas: espaco play/pausa, <-/-> +-10 s, R reiniciar.
"""
from __future__ import annotations

import argparse
import os

from swarm_sim.refuel.env import RefuelConfig, RefuelEnv, Scenario
from swarm_sim.refuel.replay import make_policy, record, save_gif, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission")
    ap.add_argument("--policy", default="hungaro_prioridade")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--contact-radius", type=float, help="sobrescreve refuel.contact_radius da missao [m]")
    ap.add_argument("--speed", type=float, default=10.0, help="segundos simulados por segundo de replay")
    ap.add_argument("--save", help="grava .gif/.mp4 em vez de abrir a janela")
    a = ap.parse_args()

    sc = Scenario.from_file(a.mission, a.synthetic)
    if a.contact_radius is not None:
        sc.m.refuel.contact_radius = a.contact_radius
    env = RefuelEnv(sc, RefuelConfig(), seed=a.seed)
    policy = make_policy(a.policy)
    frames, ret = record(env, policy)
    s = env.summary()
    print(f"{a.policy} seed {a.seed}: {s['refueled']}/{s['n_aircraft']} reabastecidas, {s['crashed']} caidas "
          f"({s['crashed_priority']} em zona prioritaria), drones sem autonomia {s['drones_failed']}, "
          f"trocas de alvo {s['retasks']}, retorno {ret:.2f}")
    title = f"{sc.name} | politica {os.path.basename(a.policy)} | seed {a.seed}"
    judged = hasattr(policy, "classify")
    if a.save:
        save_gif(env, frames, a.save, title, judged, a.speed)
        print(f"salvo em {a.save}")
    else:
        show(env, frames, title, judged, a.speed)


if __name__ == "__main__":
    main()
