"""Redes das politicas aprendidas (M24--M29): permutacao-invariantes sobre o conjunto de pistas.

A rede nao depende da ordem das pistas: cada pista vira um codigo, o contexto e a media dos codigos validos e a nota
de cada pista vem de [codigo, contexto, P(legitima)]. 'Esperar'/'manter' tem nota propria. Uma cabeca separada
classifica cada pista como legitima ou falso alarme so com caracteristicas de radar que independem do drone (``lf``);
ela e treinada com o rotulo verdadeiro da simulacao (perda auxiliar BCE) e sua probabilidade alimenta a escolha do
alvo. Como o tamanho do problema (aeronaves, drones, bases, radares) varia entre cenarios, a mesma rede serve a todos.

* ``AllocNet``  DeepSets (Zaheer et al., 2017);
* ``AttnNet``   duas camadas de codificador Transformer antes da media (Kool et al., 2019).
O treino (DAgger e PPO) esta em ``treino.py``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from swarm_sim.refuel.env import GLOB_FEATS, LEGIT_FEATS, TRACK_FEATS, RefuelEnv

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


class AttnNet(AllocNet):
    """Variante com auto-atencao entre as pistas (codificador Transformer, como em Kool et al., 2019) no lugar da
    media simples: cada pista ve as outras antes de receber a nota. Mesma interface e mesmas cabecas."""

    def __init__(self, hidden: int = 96, heads: int = 4, layers: int = 2):
        super().__init__(hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.0, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, ac, lf, glob, mask):
        B, K, _ = ac.shape
        lg = self.classify(lf)
        h = self.enc(torch.cat([ac, lf, glob[:, None, :].expand(B, K, -1)], -1))
        pad = ~mask[:, :-1]
        pad = pad.clone()
        pad[pad.all(1), 0] = False                       # linha sem pistas: evita atencao sobre conjunto vazio
        h = self.attn(h, src_key_padding_mask=pad)
        m = mask[:, :-1].float()[..., None]
        ctx = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        s = self.score(torch.cat([h, ctx[:, None, :].expand(B, K, -1), torch.sigmoid(lg)[..., None]], -1)).squeeze(-1)
        hold = self.hold(torch.cat([ctx, glob], -1))
        logits = torch.cat([s, hold], -1).masked_fill(~mask, -1e9)
        return logits, self.value(torch.cat([ctx, glob], -1)).squeeze(-1), lg


NET_KINDS = {"deepsets": AllocNet, "attn": AttnNet}


def make_net(kind: str = "deepsets") -> AllocNet:
    return NET_KINDS[kind]()


def net_kind(net: AllocNet) -> str:
    return "attn" if isinstance(net, AttnNet) else "deepsets"


def load_net(path: str, kind: str | None = None) -> AllocNet:
    """Rede salva (state_dict); o tipo e deduzido das chaves se nao for dado."""
    sd = torch.load(path, map_location="cpu")
    net = make_net(kind or ("attn" if any(k.startswith("attn.") for k in sd) else "deepsets"))
    net.load_state_dict(sd)
    net.eval()
    return net


def tensors(obs_list):
    return tuple(torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in ("ac", "lf", "glob", "mask"))


def labels(obs_list):
    return torch.as_tensor(np.stack([o["legit"] for o in obs_list]))


def legit_loss(lg, mask, y):
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
        logits, _, _ = net(*tensors([env.obs()]))
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
