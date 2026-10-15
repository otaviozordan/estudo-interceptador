"""Elementos comuns a todos os metodos (Secao 2.4 do artigo, E1 a E4) e a montagem da rodada de decisao.

* (E1) filtro de plausibilidade: ``candidates`` / ``rule_classify`` (envelope de voo ``env.plausible``);
* (E2) custo de interceptacao com histerese: ``custo`` (tempo previsto; KEEP_GAIN no alvo atual de um drone em voo);
* (E3) peso de prioridade e tempo ate a zona: ``priority_weight``, ``time_to_zone``;
* (E4) modelo de sucesso e objetivo de valor esperado: ``success``, ``wta_setup``, ``objective``.

Todas usam so o quadro de radar (pistas) e as zonas prioritarias da missao; nenhum metodo ve a posicao verdadeira.
"""
from __future__ import annotations

import numpy as np

from swarm_sim.refuel.env import RefuelEnv, intercept_time

BIG = 1e6               # custo de par inviavel (fora da autonomia)
KEEP_GAIN = 0.7         # histerese (E2): manter o alvo atual custa 70 % do tempo
P_TERM = 0.85           # (E4) probabilidade de contato dado que o drone chega ao sensor (experimento 1x1)
TAU = 600.0             # (E4) decaimento de q com o tempo de interceptacao [s]
LAUNCH_COST = 0.05      # (E4) custo de lancar um drone da base (fracao do valor de uma pista comum)


# ------------------------------------------------------------------------------------------ pistas e custo
def slot_of(env: RefuelEnv, track_id: int) -> int:
    for i, s in enumerate(env.slots):
        if s is not None and s["id"] == track_id:
            return i
    return env.cfg.max_tracks


def custo(env: RefuelEnv, drones: list[int], tracks: list[dict]) -> np.ndarray:
    """(E2) Tempo de interceptacao (BIG se inviavel pela autonomia) de cada drone a cada pista."""
    c = np.full((len(drones), len(tracks)), BIG)
    if not tracks:
        return c
    pos = np.array([[p["n"], p["e"]] for p in tracks])
    vel = np.array([[p["vn"], p["ve"]] for p in tracks])
    ids = np.array([p["id"] for p in tracks])
    for i, d in enumerate(drones):
        t = intercept_time(pos - env.dr_pos[d], vel, env.rf.drone_speed)
        ok = t <= env.rf.endurance - env.dr_fly[d]
        c[i, ok] = t[ok]
        if env.airborne(d):
            c[i] = np.where((ids == env.dr_track[d]) & ok, KEEP_GAIN * c[i], c[i])
    return c


def time_to_ground(env: RefuelEnv, p: dict) -> float:
    """Tempo ate o solo estimado so pelo radar (AGL / razao de descida); inf se nao esta descendo."""
    agl = env._agl(p)
    return agl / -p["vh"] if p["vh"] < -0.5 else np.inf


def priority_weight(env: RefuelEnv, p: dict) -> float:
    """(E3) Peso da pista: penalidade forte se ja esta dentro (ou indo para dentro) de uma zona prioritaria."""
    sd, close, inside = env.sc.zone_geometry((p["n"], p["e"]), (p["vn"], p["ve"]))
    pri, oth = env.rf.crash_penalty_priority, env.rf.crash_penalty_other
    if inside:
        return pri
    if close > 5.0 and sd < 3000.0:
        return 0.5 * (pri + oth)
    return oth


def time_to_zone(env: RefuelEnv, p: dict) -> float:
    """(E3) Tempo ate a fronteira da zona prioritaria mais proxima (0 se ja dentro, inf se nao se aproxima)."""
    sd, close, inside = env.sc.zone_geometry((p["n"], p["e"]), (p["vn"], p["ve"]))
    if inside:
        return 0.0
    return sd / close if close > 1.0 else np.inf


def rule_classify(env: RefuelEnv, tracks: list[dict]) -> np.ndarray:
    """(E1) Probabilidade (0/1) de cada pista ser uma aeronave, pela regra fisica."""
    return np.array([float(env.plausible(p)) for p in tracks])


def candidates(env: RefuelEnv, tracks: list[dict], filt: bool) -> list[dict]:
    """(E1) Pistas que passam no filtro de plausibilidade (todas, se ``filt`` e falso)."""
    return [p for p in tracks if env.plausible(p)] if filt else tracks


# ------------------------------------------------------------------------------------------ rodada
def rodada(env: RefuelEnv, filt: bool = True):
    """Drones da fila, pistas candidatas, custo (tempo) e quantos drones FORA da fila ja estao em cada pista."""
    drones = list(env.queue)
    fixed: dict[int, int] = {}
    for d in range(env.n_dr):
        if env.airborne(d) and d not in drones:
            fixed[int(env.dr_track[d])] = fixed.get(int(env.dr_track[d]), 0) + 1
    cap = env.cfg.max_per_track
    tr = [p for p in candidates(env, env._pic, filt) if fixed.get(p["id"], 0) < cap]
    return drones, tr, custo(env, drones, tr), fixed


def pick(env: RefuelEnv, drones, tr, assign: dict[int, int]) -> int:
    """Slot do drone atual dada a atribuicao {indice do drone na fila: indice da pista} (-1/ausente = esperar)."""
    j = assign.get(drones.index(env.current_drone), -1)
    return slot_of(env, tr[j]["id"]) if j >= 0 else env.cfg.max_tracks


def as_ids(drones, tr, assign) -> dict:
    """Atribuicao por indice (-1 = esperar) -> {drone: id da pista ou None}."""
    return {d: (tr[int(j)]["id"] if j is not None and j >= 0 else None) for d, j in zip(drones, assign)}


# ------------------------------------------------------------------------------------- objetivo WTA (E4)
def success(env: RefuelEnv, cost: np.ndarray) -> np.ndarray:
    """q_ij = P_TERM exp(-c_ij / TAU) se o drone alcanca a pista dentro da autonomia, 0 caso contrario."""
    return np.where(cost < BIG, P_TERM * np.exp(-np.minimum(cost, BIG) / TAU), 0.0)


def wta_setup(env: RefuelEnv, filt: bool = True):
    """Instancia WTA da rodada: (drones, pistas, (q, V, S, custo de lancamento, drones ja em cada pista))."""
    drones, tr, cost, fixed = rodada(env, filt)
    if not tr:
        return drones, tr, None
    q = success(env, cost)
    V = np.array([priority_weight(env, p) for p in tr])
    # sobrevivencia atual de cada pista dado os drones fora da fila (mesmo modelo de q, com a posicao atual deles)
    S = np.ones(len(tr))
    idx = {p["id"]: j for j, p in enumerate(tr)}
    others = [d for d in range(env.n_dr) if env.airborne(d) and d not in drones and int(env.dr_track[d]) in idx]
    if others:
        qo = success(env, custo(env, others, tr))
        for i, d in enumerate(others):
            j = idx[int(env.dr_track[d])]
            S[j] *= 1 - qo[i, j]
    launch = np.array([0.0 if env.airborne(d) else LAUNCH_COST for d in drones])
    count0 = np.array([fixed.get(p["id"], 0) for p in tr])
    return drones, tr, (q, V, S, launch, count0)


def objective(assign: np.ndarray, q, V, S, launch, count0, cap) -> np.ndarray:
    """Valor esperado destruido - custo de lancamento, para uma populacao de atribuicoes (P, nd) com -1 = esperar.
    Violacao do limite de drones por pista e penalizada."""
    P, nd = assign.shape
    nt = len(V)
    logs = np.zeros((P, nt))
    cnt = np.zeros((P, nt))
    rows = np.repeat(np.arange(P), nd)
    cols = assign.ravel()
    ds = np.tile(np.arange(nd), P)
    ok = cols >= 0
    lq = np.log(np.maximum(1 - q[ds[ok], cols[ok]], 1e-12))
    np.add.at(logs, (rows[ok], cols[ok]), lq)
    np.add.at(cnt, (rows[ok], cols[ok]), 1)
    val = (V * S * (1 - np.exp(logs))).sum(1)
    cost = ((assign >= 0) * launch[None, :]).sum(1)
    over = np.maximum(cnt + count0[None, :] - cap, 0).sum(1)
    return val - cost - 10.0 * over


def score(q, V, S, launch):
    """Valor esperado de designar o drone i a pista j sozinho (base da atribuicao 1-1 do WTA)."""
    return V[None, :] * S[None, :] * q - launch[:, None]
