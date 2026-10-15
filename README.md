# swarm_intercept

Alocação de interceptadores contra enxames de aeronaves evasivas sob informação de radar: simulador, os 29 métodos de
alocação do artigo (M1–M29), o conjunto de 60 cenários e os scripts que geram as tabelas do artigo (`paper/`).

Os dados de terreno (`data/dem/*.tif`, Copernicus GLO-30) não são versionados e ficam só localmente.
Rode todos os comandos **a partir da raiz do repositório**.

## Estrutura

```
rodar.py                    roda um ou mais métodos numa missão (com os modelos já treinados)
metodos/                    os 29 métodos, um arquivo por método
  base.py, comum.py         interface comum e elementos E1–E4 (filtro, custo, prioridade, modelo de sucesso)
  m01_aleatorio.py ... m23_arvore_utilidade.py
  aprendidos.py, redes.py   M24–M29 (DeepSets / atenção)
  treino.py                 GA da árvore (M23), DAgger (M24, M25), PPO (M26–M29)
  paralelo.py               episódios em paralelo
modelos/                    parâmetros treinados, uma pasta por método (M23 coef.json, M24–M29 s<semente>.pt)
swarm_sim/                  simulador: relevo, radares, rastreador IMM, aeronaves 6-DOF, missões
  refuel/                   ambiente de alocação (env.py), manobras evasivas, episódio/avaliação, replay
missions/
  realista/                 192 missões realistas (6 cidades × 32), conjunto padrão — ver docs/missoes_realistas.md
  enxame/                   os 60 cenários sintéticos da primeira versão do artigo + manifest.json
  reabastecimento/          bateria anterior (1–4 aeronaves, estudo de reabastecimento)
scripts/
  treinar.py                treina M23–M29 e grava em modelos/
  experimentos/             benchmark do artigo, análise (tabelas LaTeX), validação, guiagem 1×1, estudo legado
  missoes/                  geradores de missões, construtor gráfico, simulação 6-DOF de uma missão
paper/                      artigo (LaTeX); tabelas geradas em paper/tabelas/
docs/                       documentação; docs/referencias/ com os trabalhos de base (COBEM 2025, dissertação)
results/                    saídas dos experimentos (não versionado)
data/                       DEM e tiles de satélite (não versionados); data/osm/ = ativos do OpenStreetMap (versionado)
```

Instalação: `pip install -r requirements.txt` (Python 3.10+).

## Rodar um método numa missão

```
python rodar.py --listar                                        # os 29 métodos e quais estão treinados
python rodar.py --missao sao_paulo/real_sao_paulo_03 --metodo M7
python rodar.py --missao missions/realista/rio_de_janeiro/real_rio_de_janeiro_10.json --metodo M7 M13 M17 --execucoes 5
python rodar.py --missao <missao> --metodo todos --workers 7    # todos os disponíveis, em paralelo
python rodar.py --missao <missao> --metodo M24 --semente-treino 1
python rodar.py --missao <missao> --metodo M7 --ver             # replay na janela
python rodar.py --missao <missao> --metodo M7 --gif ep.gif      # grava o replay
```

`--missao` aceita um `.json`, o nome do cenário relativo a `missions/realista` ou `missions/enxame` (sem `.json`), uma
pasta ou um glob.
O método pode ser o id (`M7`, `7`) ou a chave usada nos resultados (`hungaro_prioridade`). A execução *k* usa a semente
10000 + *k*, a mesma do benchmark, então reproduz exatamente o episódio do artigo. Outras opções: `--guiagem`
(pn, pp, apn, ideal, mpc), `--amax`, `--saida episodios.jsonl`, `--modelos <pasta>`.

| id | método | arquivo |
|---|---|---|
| M1–M5 | aleatório, gulosos (próximo, urgente, prioridade), TEWA | `metodos/m01`…`m05` |
| M6–M9 | húngaro, húngaro ponderado, sem filtro, leilão de Bertsekas | `m06`…`m09` |
| M10–M12 | guloso global, CBAA, CBAA com comunicação local | `m10`…`m12` |
| M13–M16 | MMR, húngaro sobre WTA, MILP (HiGHS), busca local (VLSN) | `m13`…`m16` |
| M17–M20 | algoritmo genético, colônia de formigas, recozimento simulado, enxame de partículas | `m17`…`m20` |
| M21 | húngaro com reserva | `m21` |
| M22, M23 | árvores de comportamento (distância; utilidade afim, treinável) | `m22`, `m23` |
| M24–M29 | imitação, imitação+PPO e PPO, em DeepSets e atenção (treináveis) | `aprendidos.py`, `redes.py` |

Para usar um método no seu código:

```python
from metodos import obter
from swarm_sim.refuel.env import RefuelEnv, Scenario
from swarm_sim.refuel.episodio import config_artigo, run_episode

pol = obter("M13").politica()                    # treináveis carregam modelos/<id>_<chave>/
sc = Scenario.from_file("missions/enxame/taubate/7_radares/36_abastecimentos/enx_taubate_07.json")
print(run_episode(pol, sc, config_artigo(), seed=10_000)["dano"])
```

## Treinar (M23–M29)

```
python scripts/treinar.py M23                       # GA dos coeficientes da árvore com utilidade
python scripts/treinar.py M24 M25 --semente 0 1 2   # imitação do M7 (DAgger)
python scripts/treinar.py M26 M27 --semente 0 1 2   # imitação + PPO (treina M24/M25 antes, se faltar)
python scripts/treinar.py todos --semente 0 1 2     # M23–M29
```

Treina só nos cenários de treino (o `split` do manifest: nas missões realistas, Rio de Janeiro e Brasília ficam para
teste), com os parâmetros do artigo
(`metodos/treino.py: PARAMS`, sobrescrevíveis por opção). Grava em `modelos/<id>_<chave>/` e reaproveita o que já existe
(`--refazer` força). Para levar os modelos para outra máquina, copie `modelos/`.

## Benchmark do artigo

```
python scripts/experimentos/run_benchmark.py --dir results/benchmark/<pasta>          # todas as fases
python scripts/experimentos/run_benchmark.py --phase heur --methods M7 M13 --dir ...   # só alguns métodos
python scripts/experimentos/analyze_benchmark.py results/benchmark/<pasta>             # tabelas e figuras
python scripts/experimentos/check_equivalence.py                                       # validação (Tab. equivalência)
python scripts/experimentos/run_guidance.py                                            # guiagem 1×1 (Tab. guiagem)
```

Fases: `heur` (M1–M22, 300 episódios cada), `bt` (M23), `learn` (M24–M29, 3 sementes, avaliadas no teste) e
`ablation`. Cada episódio vira uma linha de `episodes.jsonl`; rodar de novo com o mesmo `--dir` **retoma**, pulando o que já
foi avaliado. Os modelos treinados no benchmark vão para `modelos/` (os mesmos usados por `rodar.py`). Custo: ~35 min por
método da fase `heur` em 8 núcleos; o benchmark completo leva cerca de dois dias.

## Missões

```
python scripts/missoes/osm_ativos.py                 # ativos reais (OpenStreetMap) das 6 cidades -> data/osm/
python scripts/missoes/make_realista.py              # 192 missões realistas em missions/realista
python scripts/missoes/make_swarm.py                 # os 60 cenários sintéticos antigos em missions/enxame
python scripts/missoes/make_battery.py               # bateria de reabastecimento em missions/reabastecimento
python scripts/missoes/mission_builder.py [missions/reabastecimento/minha.json] [--place itajuba]   # construtor
python scripts/missoes/run_mission.py <missao.json> [--duration 300] [--batch] [--resim]           # simulação 6-DOF
```

As missões realistas misturam quatro classes de ameaça (drone de ataque de longo alcance, isca, munição vagante e FPV)
com parâmetros de fontes abertas, atacando ativos reais do OpenStreetMap, com a defesa montada sem conhecer o ataque
(detalhes, hipóteses e fontes em `docs/missoes_realistas.md`). O relevo de São Paulo, Rio e Brasília usa os tiles
Copernicus `S24_W047`, `S23_W044` e `S16_W048` (baixe de `https://copernicus-dem-30m.s3.amazonaws.com/` para `data/dem/`).
Toda missão é um *preset* aleatório: pontos de rota, velocidade, altura, instante de entrada e manobras são sorteados a
cada execução. `refuel.runs` diz quantas execuções a missão tem na avaliação. Cada missão salva ganha um `.png` com relevo,
cobertura dos radares, rotas, zonas, bases e radares.

**Construtor** (Tkinter, tema escuro): clique esquerdo adiciona conforme o modo (1 Waypoint, 2 Chão, 3 Loiter, 4 Início,
5 Rajada, 6 Radar, 7 Base de drones, 8 Zona prioritária), arrastar move objetos ou o mapa, scroll dá zoom, clique direito
remove. Alterna **Elevação ↔ Satélite** (imagens Esri baixadas sob demanda para `data/tiles/`). Rota suave (spline
Catmull-Rom), velocidade por trecho, rajadas localizadas, aleatoriedade por waypoint. Ctrl+S organiza em
`missions/reabastecimento/<local>/<n>_radares/<n>_abastecimentos/<nome>.json`. Locais prontos: Itajubá, Pouso Alegre,
Campos do Jordão, Taubaté, Pindamonhangaba, Guaratinguetá, São José dos Campos.

## Simulador (`swarm_sim/`)

UAVs de asa fixa 6-DOF (quaternions, sustentação/arrasto/estol, empuxo, ISA, vento + rajada) sobre o DEM, com colisão com o
terreno. Referencial local NED (n = norte, e = leste, metros).

| módulo | conteúdo |
|---|---|
| `terrain.py` | DEM GeoTIFF → grade métrica local, altura/AGL/declividade, colisão de segmento, DEM fractal sintético |
| `aircraft.py`, `autopilot.py`, `guidance.py` | dinâmica 6-DOF (Aerosonde), autopiloto em cascata, guiagem L1/loiter/AGL |
| `routes.py`, `mission.py`, `gusts.py` | rotas, arquivo de missão JSON, rajadas localizadas |
| `radar.py` | radares (equação do radar, Swerling I, difração, varredura, vieses, falsos alarmes) + rastreador IMM |
| `refuel/env.py` | ambiente cinemático de alocação: aeronaves evasivas, interceptadores, rodadas de decisão, observação das redes |
| `refuel/evasion.py` | manobras evasivas (zigue-zague, curvas aleatórias, quebra reativa) e leis de guiagem (PP, PN, APN, MPC) |
| `refuel/episodio.py` | `config_artigo`, `run_episode`, `evaluate` |
| `refuel/replay.py` | replay/GIF de um episódio |

**Radares.** `SNR = SNR50·(R0/r)⁴·RCS·perdas`, alvo Swerling I (`Pd = Pfa^(1/(1+SNR))`), RCS dependente do aspecto,
difração em gume de faca e curvatura da Terra (raio 4/3), varredura real assíncrona, ruído ∝ 1/√SNR, vieses fixos por radar
e falsos alarmes perto da antena. O rastreador (IMM com dois modelos, associação por Mahalanobis + húngaro, confirmação com 3
plots, descarte após 20 s, fusão de duplicatas) nunca usa a identidade verdadeira. Os métodos de alocação só veem as pistas
(`RadarNetwork.picture`) e as zonas prioritárias; a posição verdadeira só decide contato, impacto e métricas.

**Alocação.** Interceptadores descartáveis (1 drone ↔ 1 aeronave) saem de bases; a aeronave não interceptada cai no último
ponto da rota, com penalidade forte em zona prioritária. A cada 2 s há uma rodada de decisão (drones em voo cuja
reavaliação venceu, depois os ociosos nas bases); cada um recebe uma pista ou a ação nula. Em voo o drone segue o ponto
previsto de encontro da pista e, quando a aeronave entra no sensor próprio (300 m), passa à guiagem terminal (PN, N' = 4).
Métrica principal: dano normalizado (fração da penalidade máxima).

Imagens de satélite: © Esri, Maxar, Earthstar Geographics.
