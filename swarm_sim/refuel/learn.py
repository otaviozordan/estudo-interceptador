"""Politicas neurais para a alocacao: rede permutacao-invariante, imitacao (DAgger) e PPO.

A rede nao depende da ordem das pistas: cada pista vira um embedding, o contexto e a media dos embeddings
validos e o logit de cada pista vem de [embedding, contexto, P(legitima)]. 'Esperar'/'manter' tem logit proprio.
Uma cabeca separada classifica cada pista como legitima ou falso alarme so com caracteristicas de radar que
independem do drone (``lf``); ela e treinada com o rotulo verdadeiro da simulacao (perda auxiliar BCE, em IL e PPO)
e sua probabilidade alimenta a escolha do alvo. Como o tamanho
do problema (aeronaves, drones, bases, radares) varia entre cenarios, a mesma rede serve para todos.
Treina-se sobre um conjunto de cenarios (missions/*.json), sorteando um cenario e uma semente a cada episodio.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import BASELINES
from .env import GLOB_FEATS, LEGIT_FEATS, TRACK_FEATS, RefuelConfig, RefuelEnv, Scenario

LEGIT_COEF = 0.5        # peso da perda auxiliar de legitimidade


class AllocNet(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        h = hidden
        self.legit = nn.Sequential(nn.Linear(LEGIT_FEATS, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.enc = nn.Sequential(nn.Linear(TRACK_FEATS + LEGIT_FEATS + GLOB_FEATS, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * h + 1, h), nn.ReLU(), nn.Linear(h, 1))
        self.hold = nn.Sequential(nn.Linear(h + GLOB_FEATS, h), nn.ReLU(), nn.Linear(h, 1))
        self.value = nn.Sequential(nn.Linear(h + GLOB_FEATS, h), nn.ReLU(), nn.Linear(h, 1))

    def classify(self, lf):
        """lf (..., LEGIT_FEATS) -> logit de a pista ser uma aeronave (...)."""
        return self.legit(lf).squeeze(-1)

    def forward(self, ac, lf, glob, mask):
        """ac (B,K,F), lf (B,K,L), glob (B,G), mask (B,K+1) -> logits (B,K+1) mascarados, valor (B,),
        logit de legitimidade por pista (B,K)."""
        B, K, _ = ac.shape
        lg = self.classify(lf)
        h = self.enc(torch.cat([ac, lf, glob[:, None, :].expand(B, K, -1)], -1))
        m = mask[:, :-1].float()[..., None]
        ctx = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        s = self.score(torch.cat([h, ctx[:, None, :].expand(B, K, -1), torch.sigmoid(lg)[..., None]], -1)).squeeze(-1)
        hold = self.hold(torch.cat([ctx, glob], -1))
        logits = torch.cat([s, hold], -1).masked_fill(~mask, -1e9)
        return logits, self.value(torch.cat([ctx, glob], -1)).squeeze(-1), lg


def _tensors(obs_list):
    return tuple(torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in ("ac", "lf", "glob", "mask"))


def _labels(obs_list):
    return torch.as_tensor(np.stack([o["legit"] for o in obs_list]))


def _legit_loss(lg, mask, y):
    """BCE da legitimidade nos slots validos (classes balanceadas: falsos alarmes sao maioria)."""
    m = mask[:, :-1]
    if not m.any():
        return lg.sum() * 0.0
    yv, lv = y[m], lg[m]
    pos = yv.mean().clamp(0.02, 0.98)
    w = torch.where(yv > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    return (F.binary_cross_entropy_with_logits(lv, yv, reduction="none") * w).mean()


def net_policy(net: AllocNet, greedy: bool = True):
    net.eval()

    @torch.no_grad()
    def policy(env: RefuelEnv) -> int:
        logits, _, _ = net(*_tensors([env.obs()]))
        if greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())

    @torch.no_grad()
    def classify(env: RefuelEnv, tracks: list[dict]) -> np.ndarray:
        """P(pista legitima) para pistas quaisquer do quadro de radar."""
        if not tracks:
            return np.zeros(0)
        return torch.sigmoid(net.classify(torch.as_tensor(env.legit_feats(tracks)))).numpy()
    policy.classify = classify
    return policy


def _pick(scenarios: list[Scenario], seed: int) -> Scenario:
    """Cenario do episodio, com peso = refuel.runs da missao."""
    w = np.array([max(sc.m.refuel.runs, 1) for sc in scenarios], float)
    return scenarios[int(np.random.default_rng(seed).choice(len(scenarios), p=w / w.sum()))]


def dagger_episode(env: RefuelEnv, pol, expert) -> tuple[list[dict], list[int]]:
    """O aluno (ou o especialista) dirige; o especialista rotula cada decisao."""
    obs, acts = [], []
    while not env.done:
        obs.append(env.obs())
        acts.append(expert(env))
        env.step(pol(env))
    return obs, acts


@torch.no_grad()
def ppo_episode(env: RefuelEnv, net: AllocNet):
    """Episodio com amostragem da politica: (obs, acoes, logp, recompensas, valores)."""
    o_ep, a_ep, lp_ep, r_ep, v_ep = [], [], [], [], []
    o = env.obs()
    while not env.done:
        logits, v, _ = net(*_tensors([o]))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        o_ep.append(o), a_ep.append(int(a)), lp_ep.append(float(dist.log_prob(a))), v_ep.append(float(v))
        o, r, _ = env.step(int(a))
        r_ep.append(r)
    return o_ep, a_ep, lp_ep, r_ep, v_ep


# --------------------------------------------------------------------------- imitacao (DAgger)
def train_imitation(scenarios: list[Scenario], cfg: RefuelConfig, expert: str = "hungaro_prioridade", iters: int = 5,
                    episodes: int = 40, epochs: int = 8, seed: int = 0, log=print, pool=None) -> AllocNet:
    """Clona o especialista; a partir da 2a iteracao o aluno dirige e o especialista so rotula (DAgger).
    ``pool`` (parallel.EpisodePool) roda os episodios em paralelo."""
    torch.manual_seed(seed)
    net = AllocNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    data_o, data_a = [], []
    t0 = time.time()
    for it in range(iters):
        seeds = [seed * 1_000_000 + it * 10_000 + k for k in range(episodes)]
        if pool is not None:
            spec = ("base", expert) if it == 0 else ("net", net.state_dict(), False)
            res = pool.map([("dagger", pool.index(_pick(scenarios, sd)), sd, spec, expert) for sd in seeds])
        else:
            pol = BASELINES[expert] if it == 0 else net_policy(net, greedy=False)
            res = [dagger_episode(RefuelEnv(_pick(scenarios, sd), cfg, seed=sd), pol, BASELINES[expert])
                   for sd in seeds]
        for o, a in res:
            data_o += o
            data_a += a
        ac, lf, gl, mk = _tensors(data_o)
        yl = _labels(data_o)
        y = torch.as_tensor(np.array(data_a))
        net.train()
        n = len(y)
        tot = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                logits, _, lg = net(ac[idx], lf[idx], gl[idx], mk[idx])
                loss = F.cross_entropy(logits, y[idx]) + LEGIT_COEF * _legit_loss(lg, mk[idx], yl[idx])
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
              ent: float = 0.01, seed: int = 0, log=print, pool=None) -> AllocNet:
    torch.manual_seed(seed)
    net = net or AllocNet()
    opt = torch.optim.Adam(net.parameters(), lr)
    t0 = time.time()
    for u in range(updates):
        net.eval()
        seeds = [seed * 100_000_000 + u * 1000 + k for k in range(episodes)]
        if pool is not None:
            sd_net = net.state_dict()
            eps = pool.map([("ppo", pool.index(_pick(scenarios, sd)), sd, sd_net) for sd in seeds])
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
        ac, lf, gl, mk = _tensors(obs)
        yl = _labels(obs)
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
                        + LEGIT_COEF * _legit_loss(lg, mk[idx], yl[idx]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if u % 5 == 0 or u == updates - 1:
            log(f"  [PPO] upd {u + 1}/{updates}: retorno medio={np.mean(ep_ret):.2f}  passos={n}  {time.time() - t0:.0f}s")
    net.eval()
    return net
