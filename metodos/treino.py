"""Treino dos metodos com parametros ajustados a partir de dados (so nos cenarios de TREINO):

* ``train_bt``         M23: coeficientes da utilidade afim por algoritmo genetico, minimizando o dano medio;
* ``train_imitation``  M24, M25: imitacao do especialista (hungaro ponderado, M7) por DAgger (Ross et al., 2011);
* ``train_ppo``        M26--M29: PPO (Schulman et al., 2017), do zero ou refinando a rede de imitacao.

Treina-se sobre um conjunto de cenarios, sorteando um cenario e uma semente a cada episodio. ``pool``
(``paralelo.EpisodePool``) roda os episodios em paralelo.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from swarm_sim.refuel.env import RefuelConfig, RefuelEnv, Scenario

from .redes import LEGIT_COEF, AllocNet, labels, legit_loss, make_net, net_kind, net_policy, tensors


# conjunto de missoes padrao e locais de teste do conjunto sintetico antigo (missions/enxame, sem "split" no manifest)
MISSOES_PADRAO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "missions", "realista")
LOCAIS_TESTE = ["Itajuba (centro)", "Guaratingueta"]


def split_de(meta: dict, test_places: list[str] | None = None) -> str:
    """"treino" ou "teste": o "split" gravado no manifest (missoes realistas) ou, sem ele, pelo local."""
    if test_places is None and meta.get("split"):
        return meta["split"]
    return "teste" if meta.get("local") in set(test_places or LOCAIS_TESTE) else "treino"

# parametros do artigo (Secao 4.9 e 4.10)
PARAMS = dict(expert="hungaro_prioridade", il_iters=5, il_episodes=28, il_keep=0.1, ppo_updates=40, ppo_episodes=7,
              bt_pop=12, bt_gens=8, bt_scenarios=10, bt_runs=1, bt_seed=0)


def treinar(metodo, train_scs: list[Scenario], cfg: RefuelConfig, pool, semente: int = 0, modelos: str | None = None,
            log=print, refazer: bool = False, **params) -> str:
    """Treina o metodo (M23--M29) nos cenarios de treino e grava em modelos/<pasta>/. Reaproveita o que ja existe
    (inclusive a rede de imitacao que M26/M27 refinam), a menos que ``refazer``. Devolve o caminho gravado."""
    import json
    import os
    from .aprendidos import RECEITA
    from .base import MODELOS
    from . import obter
    p = PARAMS | params
    modelos = modelos or MODELOS
    path = metodo.arquivo(semente, modelos)
    if os.path.exists(path) and not refazer:
        log(f"  {metodo.id} ja treinado: {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if metodo.treino == "coef":
        sub = train_scs[:: max(1, len(train_scs) // p["bt_scenarios"])][: p["bt_scenarios"]]
        log(f"-- {metodo.id}: GA {p['bt_pop']}x{p['bt_gens']} sobre {len(sub)} cenarios de treino")
        out = train_bt(sub, pool, p["bt_pop"], p["bt_gens"], p["bt_runs"], p["bt_seed"], log)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return path
    kind, (tipo, base) = metodo.rede, RECEITA[metodo.id]
    if tipo == "il":
        log(f"-- {metodo.id} semente {semente}: imitacao (DAgger) de {p['expert']}")
        net = train_imitation(train_scs, cfg, expert=p["expert"], iters=p["il_iters"], episodes=p["il_episodes"],
                              seed=semente, log=log, pool=pool, kind=kind, keep=p["il_keep"])
    elif base is not None:
        from .redes import load_net
        warm = load_net(treinar(obter(base), train_scs, cfg, pool, semente, modelos, log, **params), kind)
        log(f"-- {metodo.id} semente {semente}: PPO a partir de {base}")
        net = train_ppo(train_scs, cfg, warm, updates=p["ppo_updates"], episodes=p["ppo_episodes"],
                        seed=100 + semente, lr=1e-4, log=log, pool=pool)
    else:
        log(f"-- {metodo.id} semente {semente}: PPO do zero")
        net = train_ppo(train_scs, cfg, updates=p["ppo_updates"], episodes=p["ppo_episodes"], seed=200 + semente,
                        log=log, pool=pool, kind=kind)
    torch.save(net.state_dict(), path)
    return path


def _pick(scenarios: list[Scenario], seed: int) -> Scenario:
    """Cenario do episodio, com peso = refuel.runs da missao."""
    w = np.array([max(sc.m.refuel.runs, 1) for sc in scenarios], float)
    return scenarios[int(np.random.default_rng(seed).choice(len(scenarios), p=w / w.sum()))]


# ------------------------------------------------------------------------- M23: arvore com utilidade afim (GA)
def train_bt(scenarios: list[Scenario], pool, pop: int = 12, gens: int = 8, runs: int = 1, seed: int = 0,
             log=print) -> dict:
    """GA (torneio, cruzamento aritmetico, mutacao gaussiana de 10 % da faixa, elitismo por truncamento) sobre
    (c1..c4), maximizando -dano medio em ``scenarios`` (sementes 50_000 + k). Devolve {coef, dano_treino, cenarios}."""
    from .m23_arvore_utilidade import BT_COEF, COEF_HI, COEF_LO
    rng = np.random.default_rng(seed)
    lo, hi = np.array(COEF_LO), np.array(COEF_HI)

    def fitness(P):
        tasks = [("eval", pool.index(sc), 50_000 + k, ("bt", tuple(map(float, x))), None)
                 for x in P for sc in scenarios for k in range(runs)]
        rows = pool.map(tasks)
        n = len(scenarios) * runs
        return np.array([-np.mean([r["dano"] for r in rows[i * n:(i + 1) * n]]) for i in range(len(P))])

    P = rng.uniform(lo, hi, size=(pop, 4))
    P[0] = BT_COEF
    f = fitness(P)
    for g in range(gens):
        idx = rng.integers(len(P), size=(len(P), 2))
        par = P[np.where(f[idx[:, 0]] > f[idx[:, 1]], idx[:, 0], idx[:, 1])]
        mate = par[rng.permutation(len(par))]
        w = rng.random((len(P), 1))
        child = np.clip(w * par + (1 - w) * mate + rng.normal(0, 0.1, P.shape) * (hi - lo), lo, hi)
        cf = fitness(child)
        allP, allf = np.vstack([P, child]), np.concatenate([f, cf])
        keep = np.argsort(-allf)[: len(P)]
        P, f = allP[keep], allf[keep]
        log(f"  [BT-GA] ger {g + 1}/{gens}: melhor dano={-f[0]:.4f} c={np.round(P[0], 2).tolist()}")
    return {"coef": [float(c) for c in P[0]], "dano_treino": float(-f[0]), "cenarios": [sc.name for sc in scenarios]}


# ------------------------------------------------------------------------------------ episodios
def dagger_episode(env: RefuelEnv, pol, expert, keep: float = 1.0) -> tuple[list[dict], list[int]]:
    """O aluno (ou o especialista) dirige; o especialista rotula as decisoes. Cenarios com dezenas de aeronaves tem
    milhares de decisoes por episodio, quase todas 'esperar' (drones ociosos na base): guarda todas as decisoes de
    lancar/trocar e so a fracao ``keep`` das de esperar/manter, para a rede nao aprender a esperar sempre."""
    obs, acts = [], []
    rng = np.random.default_rng(env.seed + 7)
    K = env.cfg.max_tracks
    while not env.done:
        a = expert(env)
        if a != K or keep >= 1.0 or rng.random() < keep:
            o = env.obs()
            o["w"] = np.float32(1.0 if a != K or keep >= 1.0 else 1.0 / keep)   # peso de importancia
            obs.append(o)
            acts.append(a)
        env.step(a if pol is expert else pol(env))
    return obs, acts


@torch.no_grad()
def ppo_episode(env: RefuelEnv, net: AllocNet):
    """Episodio com amostragem da politica: (obs, acoes, logp, recompensas, valores)."""
    o_ep, a_ep, lp_ep, r_ep, v_ep = [], [], [], [], []
    o = env.obs()
    while not env.done:
        logits, v, _ = net(*tensors([o]))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        o_ep.append(o), a_ep.append(int(a)), lp_ep.append(float(dist.log_prob(a))), v_ep.append(float(v))
        o, r, _ = env.step(int(a))
        r_ep.append(r)
    return o_ep, a_ep, lp_ep, r_ep, v_ep


# --------------------------------------------------------------------------- imitacao (DAgger)
def train_imitation(scenarios: list[Scenario], cfg: RefuelConfig, expert: str = "hungaro_prioridade", iters: int = 5,
                    episodes: int = 40, epochs: int = 8, seed: int = 0, log=print, pool=None,
                    kind: str = "deepsets", keep: float = 1.0) -> AllocNet:
    """Clona o especialista; a partir da 2a iteracao o aluno dirige e o especialista so rotula (DAgger)."""
    from . import METHODS
    torch.manual_seed(seed)
    net = make_net(kind)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    data_o, data_a = [], []
    t0 = time.time()
    for it in range(iters):
        seeds = [seed * 1_000_000 + it * 10_000 + k for k in range(episodes)]
        if pool is not None:
            spec = ("base", expert) if it == 0 else ("net", net.state_dict(), False, kind)
            res = pool.map([("dagger", pool.index(_pick(scenarios, sd)), sd, spec, expert, keep) for sd in seeds])
        else:
            pol = METHODS[expert] if it == 0 else net_policy(net, greedy=False)
            res = [dagger_episode(RefuelEnv(_pick(scenarios, sd), cfg, seed=sd), pol, METHODS[expert], keep)
                   for sd in seeds]
        for o, a in res:
            data_o += o
            data_a += a
        ac, lf, gl, mk = tensors(data_o)
        yl = labels(data_o)
        y = torch.as_tensor(np.array(data_a))
        wt = torch.as_tensor(np.array([o.get("w", 1.0) for o in data_o], np.float32))
        net.train()
        n = len(y)
        tot = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                logits, _, lg = net(ac[idx], lf[idx], gl[idx], mk[idx])
                ce = F.cross_entropy(logits, y[idx], reduction="none")
                loss = (ce * wt[idx]).sum() / wt[idx].sum() + LEGIT_COEF * legit_loss(lg, mk[idx], yl[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
        log(f"  [IL] iter {it + 1}/{iters}: {n} amostras, loss={tot / n:.4f}  {time.time() - t0:.0f}s")
    net.eval()
    return net


# ------------------------------------------------------------------------------------- PPO
def train_ppo(scenarios: list[Scenario], cfg: RefuelConfig, net: AllocNet | None = None, updates: int = 60,
              episodes: int = 8, gamma: float = 0.995, lam: float = 0.95, clip: float = 0.2, lr: float = 3e-4,
              ent: float = 0.01, seed: int = 0, log=print, pool=None, kind: str = "deepsets") -> AllocNet:
    """PPO com GAE; ``net`` dada = refino (M26, M27), senao do zero (M28, M29)."""
    torch.manual_seed(seed)
    net = net or make_net(kind)
    kind = net_kind(net)
    opt = torch.optim.Adam(net.parameters(), lr)
    t0 = time.time()
    for u in range(updates):
        net.eval()
        seeds = [seed * 100_000_000 + u * 1000 + k for k in range(episodes)]
        if pool is not None:
            sd_net = net.state_dict()
            eps = pool.map([("ppo", pool.index(_pick(scenarios, sd)), sd, sd_net, kind) for sd in seeds])
        else:
            eps = [ppo_episode(RefuelEnv(_pick(scenarios, sd), cfg, seed=sd), net) for sd in seeds]
        obs, acts, logps, adv_all, ret_all, ep_ret = [], [], [], [], [], []
        for o_ep, a_ep, lp_ep, r_ep, v_ep in eps:
            ep_ret.append(sum(r_ep))
            adv, last, v_next = np.zeros(len(r_ep)), 0.0, 0.0
            for i in reversed(range(len(r_ep))):
                delta = r_ep[i] + gamma * v_next - v_ep[i]
                last = delta + gamma * lam * last
                adv[i], v_next = last, v_ep[i]
            obs.extend(o_ep), acts.extend(a_ep), logps.extend(lp_ep)
            adv_all.append(adv), ret_all.append(adv + np.array(v_ep))
        ac, lf, gl, mk = tensors(obs)
        yl = labels(obs)
        a_t = torch.as_tensor(acts)
        lp_old = torch.as_tensor(logps, dtype=torch.float32)
        adv = torch.as_tensor(np.concatenate(adv_all), dtype=torch.float32)
        ret = torch.as_tensor(np.concatenate(ret_all), dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        net.train()
        n = len(a_t)
        for _ in range(4):
            perm = torch.randperm(n)
            for i in range(0, n, 512):
                idx = perm[i:i + 512]
                logits, v, lg = net(ac[idx], lf[idx], gl[idx], mk[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(a_t[idx]) - lp_old[idx])
                pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]).mean()
                loss = (pg + 0.5 * F.mse_loss(v, ret[idx]) - ent * dist.entropy().mean()
                        + LEGIT_COEF * legit_loss(lg, mk[idx], yl[idx]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if u % 5 == 0 or u == updates - 1:
            log(f"  [PPO] upd {u + 1}/{updates}: retorno medio={np.mean(ep_ret):.2f}  passos={n}  {time.time() - t0:.0f}s")
    net.eval()
    return net
