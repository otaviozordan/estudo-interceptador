"""Politicas classicas de alocacao drone -> pista de radar. Interface: policy(env) -> slot (K = esperar).

Todas usam so o quadro de radar (`env.free_tracks()`) e as zonas prioritarias da missao."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .env import RefuelConfig, RefuelEnv, Scenario, intercept_time

BIG = 1e6


def _slot_of(env: RefuelEnv, track_id: int) -> int:
    for i, s in enumerate(env.slots):
        if s is not None and s["id"] == track_id:
            return i
    return env.cfg.max_tracks


def _cost(env: RefuelEnv, drones: list[int], tracks: list[dict]) -> np.ndarray:
    """Tempo de interceptacao (BIG se inviavel pela autonomia) de cada drone a cada pista."""
    c = np.full((len(drones), len(tracks)), BIG)
    if not tracks:
        return c
    pos = np.array([[p["n"], p["e"]] for p in tracks])
    vel = np.array([[p["vn"], p["ve"]] for p in tracks])
    for i, d in enumerate(drones):
        t = intercept_time(pos - env.dr_pos[d], vel, env.rf.drone_speed)
        ok = t <= env.rf.endurance - env.dr_fly[d]
        c[i, ok] = t[ok]
    return c


def _time_to_ground(env: RefuelEnv, p: dict) -> float:
    """Tempo ate o solo estimado so pelo radar (AGL / razao de descida); inf se nao esta descendo."""
    agl = p["alt"] - float(env.sc.terrain.height(p["n"], p["e"]))
    return agl / -p["vh"] if p["vh"] < -0.5 else np.inf


def _priority_weight(env: RefuelEnv, p: dict) -> float:
    """Peso da pista: penalidade forte se ja esta dentro (ou indo para dentro) de uma zona prioritaria."""
    sd, close, inside = env.sc.zone_geometry((p["n"], p["e"]), (p["vn"], p["ve"]))
    pri, oth = env.rf.crash_penalty_priority, env.rf.crash_penalty_other
    if inside:
        return pri
    if close > 5.0 and sd < 3000.0:
        return 0.5 * (pri + oth)
    return oth


def random_policy(env: RefuelEnv) -> int:
    ok = [i for i, s in enumerate(env.slots) if s is not None]
    return int(env.rng.choice(ok)) if ok else env.cfg.max_tracks


def greedy_nearest(env: RefuelEnv) -> int:
    """O drone atual vai a pista que alcança mais rapido."""
    tr = env.free_tracks()
    cost = _cost(env, [env.current_drone], tr)[0]
    if not tr or cost.min() >= BIG:
        return env.cfg.max_tracks
    return _slot_of(env, tr[int(np.argmin(cost))]["id"])


def greedy_urgent(env: RefuelEnv) -> int:
    """Pista que parece estar mais perto de cair (radar: AGL/descida) entre as alcancaveis."""
    tr = env.free_tracks()
    cost = _cost(env, [env.current_drone], tr)[0]
    best, key = None, np.inf
    for p, c in zip(tr, cost):
        k = _time_to_ground(env, p) - c
        if c < BIG and k < key:
            best, key = p, k
    return greedy_nearest(env) if best is None else _slot_of(env, best["id"])


def greedy_priority(env: RefuelEnv) -> int:
    """Minimiza tempo / peso da zona: pistas em zona prioritaria tem preferencia."""
    tr = env.free_tracks()
    cost = _cost(env, [env.current_drone], tr)[0]
    if not tr or cost.min() >= BIG:
        return env.cfg.max_tracks
    w = np.array([_priority_weight(env, p) for p in tr])
    score = np.where(cost < BIG, cost / w, np.inf)
    return _slot_of(env, tr[int(np.argmin(score))]["id"])


def _hungarian(env: RefuelEnv, weighted: bool) -> int:
    drones, tr = env._pending(), env.free_tracks()
    if not tr:
        return env.cfg.max_tracks
    cost = _cost(env, drones, tr)
    if weighted:
        cost = np.where(cost < BIG, cost / np.array([_priority_weight(env, p) for p in tr])[None, :], BIG)
    rows, cols = linear_sum_assignment(cost)
    for r, k in zip(rows, cols):
        if drones[r] == env.current_drone and cost[r, k] < BIG:
            return _slot_of(env, tr[k]["id"])
    return env.cfg.max_tracks


def hungarian(env: RefuelEnv) -> int:
    """Atribuicao otima conjunta (drones ociosos x pistas livres) minimizando a soma dos tempos."""
    return _hungarian(env, weighted=False)


def hungarian_priority(env: RefuelEnv) -> int:
    """Idem, com tempos ponderados pelo peso da zona prioritaria."""
    return _hungarian(env, weighted=True)


BASELINES = {"aleatorio": random_policy, "guloso_proximo": greedy_nearest, "guloso_urgente": greedy_urgent,
             "guloso_prioridade": greedy_priority, "hungaro": hungarian, "hungaro_prioridade": hungarian_priority}


def evaluate(policy, scenarios: list[Scenario], cfg: RefuelConfig, episodes_per_scenario: int = 5,
             seed0: int = 10_000) -> dict:
    """Mesmas sementes/cenarios para todos os metodos. Devolve medias globais e por cenario."""
    rows, per = [], {}
    for sc in scenarios:
        for k in range(episodes_per_scenario):
            env = RefuelEnv(sc, cfg, seed=seed0 + k)
            ret = 0.0
            while not env.done:
                _, r, _ = env.step(policy(env))
                ret += r
            s = env.summary() | {"return": ret}
            rows.append(s)
            per.setdefault(sc.name, []).append(s)
    keys = ["rate", "crashed", "crashed_priority", "penalty", "success", "mean_t", "drones_failed", "return"]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out["por_cenario"] = {n: {k: float(np.nanmean([r[k] for r in v])) for k in keys} for n, v in per.items()}
    return out
