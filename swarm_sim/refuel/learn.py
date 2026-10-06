"""Politicas neurais para a alocacao: rede permutacao-invariante, imitacao (DAgger) e PPO.

A rede nao depende da ordem das pistas: cada pista vira um embedding, o contexto e a media dos embeddings
validos e o logit de cada pista vem de [embedding, contexto]. 'Esperar' tem logit proprio. Como o tamanho
do problema (aeronaves, drones, bases, radares) varia entre cenarios, a mesma rede serve para todos.
Treina-se sobre um conjunto de cenarios (missions/*.json), sorteando um cenario e uma semente a cada episodio.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import hungarian_priority
from .env import GLOB_FEATS, TRACK_FEATS, RefuelConfig, RefuelEnv, Scenario


class AllocNet(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        h = hidden
        self.enc = nn.Sequential(nn.Linear(TRACK_FEATS + GLOB_FEATS, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.hold = nn.Sequential(nn.Linear(h + GLOB_FEATS, h), nn.ReLU(), nn.Linear(h, 1))
        self.value = nn.Sequential(nn.Linear(h + GLOB_FEATS, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, ac, glob, mask):
        """ac (B,K,F), glob (B,G), mask (B,K+1) -> logits (B,K+1) mascarados, valor (B,)."""
        B, K, _ = ac.shape
        h = self.enc(torch.cat([ac, glob[:, None, :].expand(B, K, -1)], -1))
        m = mask[:, :-1].float()[..., None]
        ctx = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        s = self.score(torch.cat([h, ctx[:, None, :].expand(B, K, -1)], -1)).squeeze(-1)
        hold = self.hold(torch.cat([ctx, glob], -1))
        logits = torch.cat([s, hold], -1).masked_fill(~mask, -1e9)
        return logits, self.value(torch.cat([ctx, glob], -1)).squeeze(-1)


def _tensors(obs_list):
    return (torch.as_tensor(np.stack([o["ac"] for o in obs_list])),
            torch.as_tensor(np.stack([o["glob"] for o in obs_list])),
            torch.as_tensor(np.stack([o["mask"] for o in obs_list])))


def net_policy(net: AllocNet, greedy: bool = True):
    net.eval()

    @torch.no_grad()
    def policy(env: RefuelEnv) -> int:
        logits, _ = net(*_tensors([env.obs()]))
        if greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())
    return policy


def _episode_env(scenarios: list[Scenario], cfg: RefuelConfig, seed: int) -> RefuelEnv:
    rng = np.random.default_rng(seed)
    return RefuelEnv(scenarios[int(rng.integers(len(scenarios)))], cfg, seed=seed)


# --------------------------------------------------------------------------- imitacao (DAgger)
def train_imitation(scenarios: list[Scenario], cfg: RefuelConfig, expert=hungarian_priority, iters: int = 5,
                    episodes: int = 40, epochs: int = 8, seed: int = 0, log=print) -> AllocNet:
    """Clona o especialista; a partir da 2a iteracao o aluno dirige e o especialista so rotula (DAgger)."""
    torch.manual_seed(seed)
    net = AllocNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    data_o, data_a = [], []
    for it in range(iters):
        pol = expert if it == 0 else net_policy(net, greedy=False)
        for k in range(episodes):
            env = _episode_env(scenarios, cfg, seed * 1_000_000 + it * 10_000 + k)
            while not env.done:
                data_o.append(env.obs())
                data_a.append(expert(env))
                env.step(pol(env))
        ac, gl, mk = _tensors(data_o)
        y = torch.as_tensor(np.array(data_a))
        net.train()
        n = len(y)
        tot = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                logits, _ = net(ac[idx], gl[idx], mk[idx])
                loss = F.cross_entropy(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
        log(f"  [IL] iter {it + 1}/{iters}: {n} amostras, loss={tot / n:.4f}")
    return net


# ------------------------------------------------------------------------------------- PPO
def train_ppo(scenarios: list[Scenario], cfg: RefuelConfig, net: AllocNet | None = None, updates: int = 60,
              episodes: int = 8, gamma: float = 0.995, lam: float = 0.95, clip: float = 0.2, lr: float = 3e-4,
              ent: float = 0.01, seed: int = 0, log=print) -> AllocNet:
    torch.manual_seed(seed)
    net = net or AllocNet()
    opt = torch.optim.Adam(net.parameters(), lr)
    t0 = time.time()
    for u in range(updates):
        net.eval()
        obs, acts, logps, adv_all, ret_all, ep_ret = [], [], [], [], [], []
        for k in range(episodes):
            env = _episode_env(scenarios, cfg, seed * 100_000_000 + u * 1000 + k)
            o_ep, a_ep, lp_ep, r_ep, v_ep = [], [], [], [], []
            o = env.obs()
            while not env.done:
                with torch.no_grad():
                    logits, v = net(*_tensors([o]))
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                o_ep.append(o), a_ep.append(int(a)), lp_ep.append(float(dist.log_prob(a))), v_ep.append(float(v))
                o, r, _ = env.step(int(a))
                r_ep.append(r)
            ep_ret.append(sum(r_ep))
            adv, last, v_next = np.zeros(len(r_ep)), 0.0, 0.0
            for i in reversed(range(len(r_ep))):
                delta = r_ep[i] + gamma * v_next - v_ep[i]
                last = delta + gamma * lam * last
                adv[i], v_next = last, v_ep[i]
            obs.extend(o_ep), acts.extend(a_ep), logps.extend(lp_ep)
            adv_all.append(adv), ret_all.append(adv + np.array(v_ep))
        ac, gl, mk = _tensors(obs)
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
                logits, v = net(ac[idx], gl[idx], mk[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(a_t[idx]) - lp_old[idx])
                pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]).mean()
                loss = pg + 0.5 * F.mse_loss(v, ret[idx]) - ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if u % 5 == 0 or u == updates - 1:
            log(f"  [PPO] upd {u + 1}/{updates}: retorno medio={np.mean(ep_ret):.2f}  passos={n}  {time.time() - t0:.0f}s")
    return net
