"""Politicas classicas de alocacao drone -> pista de radar. Interface: policy(env) -> slot (K = esperar).

Todas usam so o quadro de radar (`env.free_tracks()`) e as zonas prioritarias da missao. Para um drone em voo,
escolher a propria pista (ou K) mantem o alvo; manter tem desconto KEEP_GAIN no custo (histerese), entao ele so
troca de pista se a nova for bem melhor ou se a dele sumiu do radar.

Legitimidade: as heuristicas (exceto o aleatorio e o *_sem_filtro) so consideram pistas que passam na regra fisica
``env.plausible`` (envelope de voo); a decisao legitima/falsa de cada metodo fica em ``policy.classify``."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .env import RefuelConfig, RefuelEnv, Scenario, intercept_time

BIG = 1e6
KEEP_GAIN = 0.7


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
    ids = np.array([p["id"] for p in tracks])
    for i, d in enumerate(drones):
        t = intercept_time(pos - env.dr_pos[d], vel, env.rf.drone_speed)
        ok = t <= env.rf.endurance - env.dr_fly[d]
        c[i, ok] = t[ok]
        if env.airborne(d):
            c[i] = np.where((ids == env.dr_track[d]) & ok, KEEP_GAIN * c[i], c[i])
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


def rule_classify(env: RefuelEnv, tracks: list[dict]) -> np.ndarray:
    """Probabilidade (0/1) de cada pista ser uma aeronave, pela regra fisica."""
    return np.array([float(env.plausible(p)) for p in tracks])


def _candidates(env: RefuelEnv, tracks: list[dict], filt: bool) -> list[dict]:
    return [p for p in tracks if env.plausible(p)] if filt else tracks


def random_policy(env: RefuelEnv) -> int:
    """Lanca drones para pistas aleatorias; em voo nunca troca de alvo."""
    if env.airborne(env.current_drone):
        return env.cfg.max_tracks
    ok = [i for i, s in enumerate(env.slots) if s is not None]
    return int(env.rng.choice(ok)) if ok else env.cfg.max_tracks


def greedy_nearest(env: RefuelEnv, filt: bool = True) -> int:
    """O drone atual vai a pista que alcança mais rapido."""
    tr = _candidates(env, env.free_tracks(env.current_drone), filt)
    cost = _cost(env, [env.current_drone], tr)[0]
    if not tr or cost.min() >= BIG:
        return env.cfg.max_tracks
    return _slot_of(env, tr[int(np.argmin(cost))]["id"])


def greedy_urgent(env: RefuelEnv, filt: bool = True) -> int:
    """Pista que parece estar mais perto de cair (radar: AGL/descida) entre as alcancaveis."""
    tr = _candidates(env, env.free_tracks(env.current_drone), filt)
    cost = _cost(env, [env.current_drone], tr)[0]
    best, key = None, np.inf
    for p, c in zip(tr, cost):
        k = _time_to_ground(env, p) - c
        if c < BIG and k < key:
            best, key = p, k
    return greedy_nearest(env, filt) if best is None else _slot_of(env, best["id"])


def greedy_priority(env: RefuelEnv, filt: bool = True) -> int:
    """Minimiza tempo / peso da zona: pistas em zona prioritaria tem preferencia."""
    tr = _candidates(env, env.free_tracks(env.current_drone), filt)
    cost = _cost(env, [env.current_drone], tr)[0]
    if not tr or cost.min() >= BIG:
        return env.cfg.max_tracks
    w = np.array([_priority_weight(env, p) for p in tr])
    score = np.where(cost < BIG, cost / w, np.inf)
    return _slot_of(env, tr[int(np.argmin(score))]["id"])


def _hungarian(env: RefuelEnv, weighted: bool, filt: bool = True) -> int:
    drones = list(env.queue)                                    # drones que ainda decidem nesta rodada
    busy = {int(env.dr_track[d]) for d in range(env.n_dr) if env.airborne(d) and d not in drones}
    tr = _candidates(env, [p for p in env._pic if p["id"] not in busy], filt)
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


def hungarian(env: RefuelEnv, filt: bool = True) -> int:
    """Atribuicao otima conjunta (drones da rodada, na base ou em voo, x pistas livres) minimizando a soma dos tempos."""
    return _hungarian(env, weighted=False, filt=filt)


def hungarian_priority(env: RefuelEnv, filt: bool = True) -> int:
    """Idem, com tempos ponderados pelo peso da zona prioritaria."""
    return _hungarian(env, weighted=True, filt=filt)


def _policy(fn, filt: bool):
    def pol(env: RefuelEnv) -> int:
        return fn(env, filt)
    if filt:
        pol.classify = rule_classify
    return pol


BASELINES = {"aleatorio": random_policy, "guloso_proximo": _policy(greedy_nearest, True),
             "guloso_urgente": _policy(greedy_urgent, True), "guloso_prioridade": _policy(greedy_priority, True),
             "hungaro": _policy(hungarian, True), "hungaro_prioridade": _policy(hungarian_priority, True),
             "hungaro_prio_sem_filtro": _policy(hungarian_priority, False)}


def legit_scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    n = tp + fp + fn + tn
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precisao": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "acuracia": (tp + tn) / max(n, 1), "amostras": n, "frac_falsas": (fp + tn) / max(n, 1)}


def run_episode(policy, sc: Scenario, cfg: RefuelConfig, seed: int) -> dict:
    """Um episodio: resumo + retorno + matriz de confusao legitima x falsa (se ``policy.classify``)."""
    env = RefuelEnv(sc, cfg, seed=seed)
    conf = np.zeros(4, int)                                     # tp, fp, fn, tn
    classify = getattr(policy, "classify", None)
    ret, t_seen = 0.0, -1.0
    while not env.done:
        if classify is not None and env.t != t_seen and env._pic:
            t_seen = env.t
            pred = classify(env, env._pic) >= 0.5
            y = np.array([p["truth"] is not None for p in env._pic])
            conf += [np.sum(pred & y), np.sum(pred & ~y), np.sum(~pred & y), np.sum(~pred & ~y)]
        _, r, _ = env.step(policy(env))
        ret += r
    return env.summary() | {"return": ret, "scenario": sc.name, "conf": conf.tolist() if classify else None}


def aggregate(rows: list[dict]) -> dict:
    """Medias globais e por cenario (+ legitimidade se houver)."""
    keys = ["rate", "crashed", "crashed_priority", "penalty", "success", "mean_t", "drones_failed", "retasks", "return"]
    per: dict = {}
    for r in rows:
        per.setdefault(r["scenario"], []).append(r)
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out["por_cenario"] = {n: {k: float(np.nanmean([r[k] for r in v])) for k in keys} for n, v in per.items()}
    if rows and rows[0]["conf"] is not None:
        out["legit"] = legit_scores(*np.sum([r["conf"] for r in rows], 0))
    return out


def evaluate(policy, scenarios: list[Scenario], cfg: RefuelConfig, episodes_per_scenario: int | None = None,
             seed0: int = 10_000) -> dict:
    """Mesmas sementes/cenarios para todos os metodos. Devolve medias globais e por cenario e, se a politica
    decide legitimidade (``policy.classify``), a matriz de confusao legitima x falsa sobre o quadro de radar."""
    return aggregate([run_episode(policy, sc, cfg, seed0 + k)
                      for sc in scenarios for k in range(episodes_per_scenario or max(sc.m.refuel.runs, 1))])
