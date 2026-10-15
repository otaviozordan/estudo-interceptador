"""Tabelas LaTeX e figuras do artigo a partir de results/benchmark/<pasta>/episodes.jsonl (+ guiagem 1x1).

    python scripts/experimentos/analyze_benchmark.py results/benchmark/<pasta> [--guidance results/guidance/<pasta>] [--paper paper]

Gera em <paper>/tabelas/: principal.tex (teste e treino), perfil.tex, tamanho.tex, ablacao.tex, guiagem.tex,
e em <paper>/figs/: dano_vs_n.pdf. Intervalos de confianca de 95 % por bootstrap com reamostragem de CENARIOS
(episodios do mesmo cenario sao correlacionados); teste de Wilcoxon pareado (mesmo cenario e semente) contra a
referencia. Redes: media sobre as sementes de treino (o desvio entre sementes vai entre parenteses).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NAMES = {
    "aleatorio": "Aleatório", "guloso_proximo": "Guloso (próximo)", "guloso_urgente": "Guloso (urgente)",
    "guloso_prioridade": "Guloso (prioridade)", "tewa": "TEWA sequencial", "cbaa": "CBAA (guloso global)",
    "hungaro": "Húngaro", "hungaro_prioridade": "Húngaro ponderado", "hungaro_prio_sem_filtro": "Húngaro pond. s/ filtro",
    "hungaro_reserva": "Húngaro c/ reserva", "mmr": "MMR", "ga": "Algoritmo genético",
    "arvore_comportamento": "Árvore (utilidade)", "arvore_distancia": "Árvore (distância)",
    "hungaro_wta": "Húngaro sobre WTA", "milp_wta": "MILP sobre WTA (HiGHS)", "leilao": "Leilão (Bertsekas)",
    "cbaa_dist": "CBAA distribuído", "cbaa_local": "CBAA c/ comunic. local", "vlsn": "Busca local (VLSN)",
    "aco": "Colônia de formigas", "sa": "Recozimento simulado", "pso": "Enxame de partículas", "PPO-ATT": "PPO (atenção)",
    "IL-DS": "Imitação (DeepSets)", "IL-ATT": "Imitação (atenção)", "IL+PPO-DS": "Imitação+PPO (DeepSets)",
    "IL+PPO-ATT": "Imitação+PPO (atenção)", "PPO-DS": "PPO (DeepSets)",
}
FAMILY = {"aleatorio": "ref.", "guloso_proximo": "gulosa", "guloso_urgente": "gulosa", "guloso_prioridade": "gulosa",
          "tewa": "TEWA", "cbaa": "distrib.", "hungaro": "exata", "hungaro_prioridade": "exata",
          "hungaro_prio_sem_filtro": "exata", "hungaro_reserva": "dinâmica", "mmr": "WTA gulosa",
          "ga": "metaheur.", "arvore_comportamento": "comportam.", "arvore_distancia": "comportam.",
          "hungaro_wta": "exata WTA", "milp_wta": "exata WTA", "leilao": "exata/distrib.", "cbaa_dist": "distrib.",
          "cbaa_local": "distrib.", "vlsn": "busca local", "aco": "metaheur.", "sa": "metaheur.", "pso": "metaheur.",
          "PPO-ATT": "reforço", "IL-DS": "imitação", "IL-ATT": "imitação", "IL+PPO-DS": "reforço",
          "IL+PPO-ATT": "reforço", "PPO-DS": "reforço"}
REF = "hungaro_prioridade"
PROFILE_NAMES = {"nenhuma": "nenhuma", "manobra": "manobra", "reativa": "reativa", "baixa": "voo baixo",
                 "mista": "mista"}
VARIANTS = {"base": "PN (base)", "base_abl": "PN (base)", "guia_ideal": "guiagem ideal", "guia_pp": "perseguição pura", "guia_pn": "PN",
            "guia_apn": "APN", "guia_mpc": "MPC", "sem_troca": "sem troca de alvo", "salva2": "salvas de 2"}


def boot_ci(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Media e IC 95 % reamostrando cenarios."""
    g = df.groupby("scenario")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(g), size=(n, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_p(df: pd.DataFrame, m: str, col: str = "dano") -> float | None:
    """Wilcoxon pareado (cenario, semente) do metodo m contra REF; redes: media sobre sementes de treino."""
    a = df[df.metodo == m].groupby(["scenario", "seed"])[col].mean()
    b = df[df.metodo == REF].groupby(["scenario", "seed"])[col].mean()
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    d = j["a"] - j["b"]
    if m == REF or len(j) < 5 or np.allclose(d, 0):
        return None
    return float(wilcoxon(j["a"], j["b"]).pvalue)


def fmt_p(p):
    if p is None:
        return "--"
    return "$<$0,001" if p < 1e-3 else f"{p:.3f}".replace(".", ",")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def order(methods):
    keys = list(NAMES)
    return sorted(methods, key=lambda m: keys.index(m) if m in keys else 99)


def main_table(df: pd.DataFrame, split: str, path: str, caption: str, label: str):
    d = df[(df.split == split) & ((df.variante == "base") | df.variante.str.match(r"^s\d+$"))]
    lines = []
    for m in order(d.metodo.unique()):
        x = d[d.metodo == m]
        mean, lo, hi = boot_ci(x, "dano")
        seeds = x[x.variante.str.match(r"^s\d+$")]
        sd = ""
        if len(seeds):
            per = seeds.groupby("variante")["dano"].mean()
            sd = f" ({num(per.std(ddof=1) if len(per) > 1 else 0.0)})"
        rate = 100 * x["rate"].mean()
        used = (x["drones_usados"] / x["n_aircraft"]).mean()
        custo = x["custo"].mean() if "custo" in x else mean               # resultados antigos: sem desperdicio
        waste = x["desperdicio"].mean() if "desperdicio" in x else float("nan")
        lines.append(f"{NAMES.get(m, m)} & {FAMILY.get(m, '')} & {num(mean)}{sd} & [{num(lo)}; {num(hi)}] & "
                     f"{num(custo)} & {num(waste, 1)} & "
                     f"{num(rate, 1)} & {num(x['crashed_priority'].mean(), 2)} & {num(used, 2)} & "
                     f"{num(x['ms_decisao'].mean(), 2)} & {fmt_p(paired_p(d, m))} \\\\")
    n_sc, n_ep = d.scenario.nunique(), len(d[d.metodo == REF])
    body = "\n".join(lines)
    tex = rf"""\begin{{table*}}[t]
\centering
\caption{{{caption} {n_sc} cenários, {n_ep} episódios por método (mesmas sementes). Dano: fração da penalidade máxima
(menor é melhor), média e IC 95\,\% por \emph{{bootstrap}} sobre cenários; nas redes, entre parênteses, o desvio-padrão entre
sementes de treino. Custo: dano somado à penalidade dos interceptadores desperdiçados (sem autonomia, gastos em iscas ou
em voo sem uso no fim), com a mesma normalização; Desperd.: interceptadores desperdiçados por episódio. $p$: Wilcoxon
pareado contra o húngaro ponderado.}}
\label{{{label}}}
\small
\begin{{tabular}}{{@{{}}llcccccccccc@{{}}}}
\toprule
Método & Família & Dano & IC 95\,\% & Custo & Desperd. & Intercept. (\%) & Quedas prior. & Drones/aeron. & ms/decisão & $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    open(path, "w", encoding="utf-8").write(tex)


def breakdown(df: pd.DataFrame, col: str, names: dict | None, path: str, caption: str, label: str):
    d = df[(df.variante == "base") & ~df.metodo.str.contains("IL|PPO")]
    cats = sorted(d[col].dropna().unique(), key=lambda v: (list(names).index(v) if names and v in names else v))
    head = " & ".join(str(names.get(c, c) if names else c) for c in cats)
    lines = []
    best = {c: d[d[col] == c].groupby("metodo")["dano"].mean().min() for c in cats}
    for m in order(d.metodo.unique()):
        cells = []
        for c in cats:
            v = d[(d.metodo == m) & (d[col] == c)]["dano"].mean()
            s = num(v)
            cells.append(rf"\textbf{{{s}}}" if np.isclose(v, best[c]) else s)
        lines.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    cols = "c" * len(cats)
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\begin{{tabular}}{{@{{}}l{cols}@{{}}}}
\toprule
Método & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    open(path, "w", encoding="utf-8").write(tex)


def ablation_table(df: pd.DataFrame, path: str):
    d = df[~df.metodo.str.contains("IL|PPO")]
    have = set(d.variante)
    vs = [v for v in VARIANTS if v in have and v not in ("base", "base_abl")]
    vs = ["base_abl" if "base_abl" in have else "base"] + vs              # base com os mesmos sorteios das ablacoes
    ms = order([m for m in d.metodo.unique() if d[d.metodo == m].variante.isin(set(vs) - {"base"}).any()])
    if not ms:
        return
    lines = []
    for m in ms:
        cells = []
        for v in vs:
            x = d[(d.metodo == m) & (d.variante == v)]
            cells.append(num(x["dano"].mean()) if len(x) else "--")
        lines.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    head = " & ".join(VARIANTS[v] for v in vs)
    tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Ablações (dano médio em todos os cenários). A coluna ``base'' usa a guiagem padrão do benchmark.}}
\label{{tab:ablacao}}
\small
\begin{{tabular}}{{@{{}}l{'c' * len(vs)}@{{}}}}
\toprule
Método & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    open(path, "w", encoding="utf-8").write(tex)


def guidance_table(gdir: str, path: str):
    g = json.load(open(os.path.join(gdir, "guiagem.json"), encoding="utf-8"))
    rows = pd.DataFrame(g["linhas"])
    amax = sorted(rows.amax.unique())
    laws = {"pp": "PP", "pn3": "PN ($N'{=}3$)", "pn4": "PN ($N'{=}4$)", "apn4": "APN ($N'{=}4$)", "mpc": "MPC"}
    if g["args"].get("realista"):
        return guidance_table_classes(g, rows, amax, laws, path)
    lines = []
    for pr in PROFILE_NAMES:
        x = rows[rows.evasao == ("zigue_zague" if pr == "manobra" else pr)] if pr != "baixa" else None
        if x is None or not len(x):
            continue
        for i, (lw, lname) in enumerate(laws.items()):
            cells = []
            for a in amax:
                r = x[(x.lei == lw) & (x.amax == a)].iloc[0]
                cells.append(f"{num(100 * r.p_1a, 0)} / {num(100 * r.p_contato, 0)}")
            first = ("zigue-zague" if pr == "manobra" else PROFILE_NAMES[pr]) if i == 0 else ""
            lines.append(f"{first} & {lname} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    for pr in ("aleatoria",):
        x = rows[rows.evasao == pr]
        for i, (lw, lname) in enumerate(laws.items()):
            cells = [f"{num(100 * x[(x.lei == lw) & (x.amax == a)].iloc[0].p_1a, 0)} / "
                     f"{num(100 * x[(x.lei == lw) & (x.amax == a)].iloc[0].p_contato, 0)}" for a in amax]
            lines.append(f"{'curvas aleatórias' if i == 0 else ''} & {lname} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    head = " & ".join(rf"$a_{{\max}}={a:.0f}$" for a in amax)
    n = int(rows.n.iloc[0])
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Guiagem terminal 1$\times$1: probabilidade (\%) de contato na primeira passagem / em até 60~s, para
{n} encontros por célula (mesmos encontros para todas as leis). Drone a 42~m/s; aeronave a 22--32~m/s com até
10~m/s$^2$ de aceleração lateral; raio de contato {g['args']['contact']:.0f}~m; $a_{{\max}}$ do drone em m/s$^2$.}}
\label{{tab:guiagem}}
\small
\begin{{tabular}}{{@{{}}ll{'c' * len(amax)}@{{}}}}
\toprule
Manobra & Lei & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    open(path, "w", encoding="utf-8").write(tex)


CLASS_NAMES = {"ataque": "ataque (longo alcance)", "isca": "isca", "vagante": "munição vagante", "fpv": "FPV",
               "fpv_reativo": "FPV reativo"}


def guidance_table_classes(g: dict, rows: pd.DataFrame, amax, laws: dict, path: str):
    """Tabela de guiagem do modo --realista: classes de ameaca x leis; ultima linha de cada lei = media das classes."""
    lines = []
    for cl in [c for c in CLASS_NAMES if c in set(rows.evasao)]:
        x = rows[rows.evasao == cl]
        for i, (lw, lname) in enumerate(laws.items()):
            cells = []
            for a in amax:
                r = x[(x.lei == lw) & (x.amax == a)].iloc[0]
                cells.append(f"{num(100 * r.p_1a, 0)} / {num(100 * r.p_contato, 0)}")
            lines.append(f"{CLASS_NAMES[cl] if i == 0 else ''} & {lname} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    body = "\n".join(lines)
    head = " & ".join(rf"$a_{{\max}}={a:.0f}$" for a in amax)
    n, a = int(rows.n.iloc[0]), g["args"]
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Guiagem terminal 1$\times$1 por classe de ameaça: probabilidade (\%) de contato na primeira passagem / em até
{a['t_max']:.0f}~s, para {n} encontros por célula (mesmos encontros para todas as leis). Interceptador a
{a['drone_speed']:.0f}~m/s, partindo a 300~m do alvo com aspecto aleatório; velocidade e manobra do alvo conforme a classe
(Tabela~\ref{{tab:classes}}); raio de contato {a['contact']:.0f}~m; $a_{{\max}}$ do interceptador em m/s$^2$.}}
\label{{tab:guiagem}}
\small
\begin{{tabular}}{{@{{}}ll{'c' * len(amax)}@{{}}}}
\toprule
Classe & Lei & {head} \\
\midrule
{body}
\end{{tabular}}
\end{{table}}
"""
    open(path, "w", encoding="utf-8").write(tex)


COMPOSICAO_NAMES = {"saturacao": "saturação", "longo_alcance": "longo alcance", "tatico": "tática",
                    "equilibrado": "equilibrada"}


def fig_size(df: pd.DataFrame, path: str):
    d = df[((df.variante == "base") | df.variante.str.match(r"^s\d+$"))]
    keep = ["guloso_proximo", "tewa", "hungaro_prioridade", "mmr", "ga", "IL-DS", "IL-ATT", "IL+PPO-DS", "IL+PPO-ATT"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for m in [k for k in keep if k in set(d.metodo)]:
        x = d[(d.metodo == m) & (d.split == "teste")].groupby("c_n_aeronaves")["dano"].mean()
        ax.plot(x.index, x.values, marker="o", ms=3, lw=1.2, label=NAMES.get(m, m),
                ls="--" if ("IL" in m or "PPO" in m) else "-")
    ax.set_xlabel("aeronaves no cenário")
    ax.set_ylabel("dano (cenários de teste)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--guidance", default=None, help="pasta de results/guidance (padrao: a mais recente)")
    ap.add_argument("--paper", default=os.path.join(ROOT, "paper"))
    a = ap.parse_args()
    tdir, fdir = os.path.join(a.paper, "tabelas"), os.path.join(a.paper, "figs")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    df = pd.read_json(os.path.join(a.dir, "episodes.jsonl"), lines=True)
    main_table(df, "teste", os.path.join(tdir, "principal_teste.tex"),
               "Resultados nos cenários de teste (cidades nunca vistas no treino: Rio de Janeiro e Brasília).", "tab:principal")
    main_table(df, "treino", os.path.join(tdir, "principal_treino.tex"),
               "Resultados nos cenários de treino (São José dos Campos, Taubaté, Itajubá e São Paulo; as redes foram treinadas neles).", "tab:treino")
    if "c_composicao" in df:                                            # missoes realistas
        breakdown(df, "c_composicao", COMPOSICAO_NAMES, os.path.join(tdir, "composicao.tex"),
                  "Dano médio por composição do enxame (todos os cenários; em negrito, o melhor de cada coluna).",
                  "tab:composicao")
        breakdown(df, "c_local", None, os.path.join(tdir, "cidade.tex"),
                  "Dano médio por cidade (Rio de Janeiro e Brasília são teste).", "tab:cidade")
    else:
        breakdown(df, "c_perfil", PROFILE_NAMES, os.path.join(tdir, "perfil.tex"),
                  "Dano médio por perfil evasivo (todos os cenários; em negrito, o melhor de cada coluna).", "tab:perfil")
    breakdown(df, "c_n_aeronaves", None, os.path.join(tdir, "tamanho.tex"),
              "Dano médio pelo número de ameaças no cenário (todos os cenários).", "tab:tamanho")
    ablation_table(df, os.path.join(tdir, "ablacao.tex"))
    fig_size(df, os.path.join(fdir, "dano_vs_n.pdf"))
    gdir = a.guidance or max(glob.glob(os.path.join(ROOT, "results", "guidance", "*")), default=None)
    if gdir:
        guidance_table(gdir, os.path.join(tdir, "guiagem.tex"))
    s = df[df.variante == "base"].groupby("metodo")[["dano", "rate", "ms_decisao"]].mean().sort_values("dano")
    print(s.to_string())
    print(f"tabelas em {tdir}/, figuras em {fdir}/")


if __name__ == "__main__":
    main()
