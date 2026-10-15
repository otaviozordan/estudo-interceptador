"""Execucao e avaliacao de episodios para qualquer politica ``policy(env) -> slot`` (os metodos estao em metodos/).

``config_artigo`` e a configuracao do ambiente usada no benchmark do artigo; ``run_episode`` roda um episodio e
devolve o resumo (dano, taxa de interceptacao, ...) com a matriz de confusao legitima x falsa quando a politica
decide legitimidade (``policy.classify``)."""
from __future__ import annotations

import time

import numpy as np

from .env import RefuelConfig, RefuelEnv, Scenario


def config_artigo(guidance: str = "pn", amax: float = 30.0) -> RefuelConfig:
    """Ambiente do benchmark: ate 64 pistas por decisao, PN com N' = 4, a_max = 30 m/s^2."""
    return RefuelConfig(max_tracks=64, t_max=3000.0, start_frac=(0.0, 0.05), guidance=guidance, drone_amax=amax,
                        pn_gain=4.0)


def legit_scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    n = tp + fp + fn + tn
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precisao": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "acuracia": (tp + tn) / max(n, 1), "amostras": n, "frac_falsas": (fp + tn) / max(n, 1)}


def run_episode(policy, sc: Scenario, cfg: RefuelConfig, seed: int) -> dict:
    """Um episodio: resumo + retorno + matriz de confusao legitima x falsa (se ``policy.classify``)."""
    env = RefuelEnv(sc, cfg, seed=seed)
    env.skip_obs = True                                         # redes chamam env.obs() por conta propria
    conf = np.zeros(4, int)                                     # tp, fp, fn, tn
    classify = getattr(policy, "classify", None)
    ret, t_seen, t_pol, n_dec = 0.0, -1.0, 0.0, 0
    while not env.done:
        if classify is not None and env.t != t_seen and env._pic:
            t_seen = env.t
            pred = classify(env, env._pic) >= 0.5
            y = np.array([p["truth"] is not None for p in env._pic])
            conf += [np.sum(pred & y), np.sum(pred & ~y), np.sum(~pred & y), np.sum(~pred & ~y)]
        t0 = time.perf_counter()
        a = policy(env)
        t_pol += time.perf_counter() - t0
        n_dec += 1
        _, r, _ = env.step(a)
        ret += r
    return env.summary() | {"return": ret, "scenario": sc.name, "conf": conf.tolist() if classify else None,
                            "seed": seed, "decisoes": n_dec, "ms_decisao": 1e3 * t_pol / max(n_dec, 1)}


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
