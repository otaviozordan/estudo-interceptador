# Missões realistas (`missions/realista/`)

Gerador: `scripts/missoes/make_realista.py`. Ativos: `scripts/missoes/osm_ativos.py` → `data/osm/<cidade>.json`.

## Conjunto

| | |
|---|---|
| Cidades | São José dos Campos, Taubaté, Itajubá, São Paulo (treino); Rio de Janeiro, Brasília (teste) |
| Missões | 32 por cidade = 4 tamanhos (12, 24, 36, 48 ameaças) × 4 composições × 2 repetições; 192 no total (128 treino, 64 teste) |
| Execuções | 5 sorteios por missão (`refuel.runs`); cada sorteio muda pontos de rota, velocidade, altura e entrada |
| Mapa | 12–14 km de lado, relevo Copernicus GLO-30 |

Treino e teste são separados por cidade: as redes e a árvore com utilidade nunca veem o relevo nem os ativos do Rio e
de Brasília. O `split` fica gravado no `manifest.json`, junto com a composição, as classes, os ativos defendidos
(com o identificador OSM), o eixo de ameaça e a cobertura de radar das rotas.

## O que é real e o que é modelado

| Elemento | Origem |
|---|---|
| Relevo | Copernicus GLO-30 (30 m) |
| Zonas prioritárias e alvos | ativos reais do OpenStreetMap: usinas, aeródromos, subestações, água e esgoto, telecom, áreas industriais ≥ 5 ha; ativos defendidos sorteados em dois passos: a categoria, com peso importância³ (4: usina, aeródromo; 3: subestação, água, indústria ≥ 50 ha; 2: telecom, indústria), e depois o ativo dentro dela, favorecendo os maiores; assim as dezenas de torres de telecom de uma cidade não dominam a escolha |
| Defesa | montada **sem conhecer o ataque**: 4–7 radares nos morros que melhor cobrem a área até 5 km de cada ativo defendido; uma base de interceptadores a 0,8–2 km de cada ativo (mais, às vezes, uma central); 1,25 interceptador por ameaça; nenhuma missão é descartada por cobertura |
| Eixo de ameaça | entradas num setor de 60–90° da borda; cada onda pode girar até 20° |
| Rotas | caminho de menor custo numa grade de 150 m: custo = 1 + β·altura normalizada + ruído suave + penalidade de borda; β = 3 para quem voa baixo (segue vales), 0,3 para quem voa alto |
| Alvos | 75 % das ameaças vão a um ativo defendido, 25 % a outros ativos (impacto com penalidade baixa); iscas vão a ativos quaisquer |
| Chegada | 1–3 ondas separadas por 120–300 s; dentro da onda, chegadas coordenadas em até 60 s; iscas chegam 30–90 s antes |

## Classes de ameaça

| Classe | Referência | Velocidade | Altura (AGL) | Mergulho | Manobra | RCS (m²) | Valor do dano |
|---|---|---|---|---|---|---|---|
| ataque | Shahed-136 / Geran-2 | 42–51 m/s (150–185 km/h) | 50 %: 60–150 m; 50 %: 500–1500 m | 45° | nenhuma (rota pré-programada) | 0,3–1,0 | 1 |
| isca | Gerbera | 27–44 m/s (~97 km/h de cruzeiro, até ~160) | 70 %: 80–200 m; 30 %: 500–1200 m | raso | nenhuma | 0,5–1,5 | 0 |
| vagante | Lancet-3 | 22–31 m/s (80–110 km/h) | 300–800 m, órbita de 400–700 m sobre o alvo | ~63° | nenhuma | 0,1–0,4 | 0,5 |
| fpv | FPV de ataque | 17–33 m/s (60–120 km/h) | 20–50 m | raso | zigue-zague leve; metade reage ao interceptador | 0,02–0,1 | 0,25 |

Composições (fração de cada classe; toda missão tem pelo menos uma de cada):

| Composição | ataque | isca | vagante | fpv |
|---|---|---|---|---|
| saturação | 45 % | 40 % | 10 % | 5 % |
| longo alcance | 60 % | 25 % | 10 % | 5 % |
| tática | 15 % | 10 % | 40 % | 35 % |
| equilibrada | 25 % | 25 % | 25 % | 25 % |

Interceptador (referência Sting): 65 m/s no engajamento (cruzeiro anunciado de 140–170 km/h, máximo de 280 km/h),
autonomia de 420 s, aceleração lateral de 30 m/s². O filtro de plausibilidade (E1) usa o envelope que cobre todas as
classes, inclusive o mergulho: 12–75 m/s, |v_z| ≤ 60 m/s, AGL de −50 a 2500 m.

**Hipóteses do modelo** (não vêm das fontes): RCS de cada classe; valor do dano (proporcional, de forma comprimida, à
carga explosiva); inclinação do mergulho do ataque; reação do FPV; proporções das composições, exceto a de
saturação, que segue os ~60 % de ataque / 40 % de iscas relatados.

## Métricas com iscas

* `dano`: penalidade das quedas de ameaças reais (iscas valem 0), normalizada pela penalidade máxima.
* `rate`: fração das ameaças **reais** interceptadas; `iscas_abatidas` conta os interceptadores gastos em iscas.
* `desperdicio`: interceptadores lançados que não neutralizaram ameaça real: acabaram a autonomia sem contato, abateram
  uma isca, ou ainda voavam sem uso no fim do episódio.
* `custo` = (penalidade dos impactos + `waste_penalty` × `desperdicio`) / penalidade máxima, com `waste_penalty` = 0,2
  (1/5 do impacto de uma ameaça plena num ativo não defendido; hipótese do modelo, ajustável em `refuel.waste_penalty`).
  Pode passar de 1. A mesma penalidade entra na recompensa do aprendizado por reforço, então as redes treinadas com PPO
  aprendem a não disparar à toa. O `dano` não inclui o desperdício, para continuar comparável.
* Sem interceptadores restantes, as ameaças ainda em voo contam como impacto (antes o episódio terminava com elas sem
  desfecho, o que subestimava o dano quando a defesa saturava).

## Limitações

* Os interceptadores são pontos de massa em 2D: a altura da ameaça só afeta o radar, não o engajamento.
* A velocidade de cada ameaça é constante na rota (o aumento de velocidade do Lancet no mergulho não é modelado).
* Os ativos dependem do mapeamento do OSM (a cobertura varia entre cidades); os arquivos de `data/osm/` fixam o
  estado do OSM na data do download.
* Mapas de 12–14 km: as rotas de longo alcance começam na borda, isto é, só o trecho final é simulado.

## Fontes

* Shahed-136 / Geran-2 (velocidade, faixa de altitude, rota pré-programada): [GUR – War & Sanctions](https://war-sanctions.gur.gov.ua/en/uav/336), [GlobalSecurity](https://www.globalsecurity.org/military/world/russia/geran-2.html), [Drone Warfare](https://drone-warfare.com/research/shahed-136/), [FAIR](https://www.fair.org.ua/en/shahed-136-harakterystyky/)
* Gerbera (papel de isca, velocidade, proporção 60/40 nas ondas, assinatura): [ISIS](https://isis-online.org/isis-reports/russian-decoy-drones-that-depend-on-western-parts-pose-a-great-challenge/), [CAPSS India](https://capssindia.org/?p=16521), [defence-network.com](https://defence-network.com/gerbera-russland-taeuschungsdrohne/), [TWZ](https://www.twz.com/air/new-russian-long-range-drone-appears-in-ukraine)
* Lancet-3 (cruzeiro, mergulho, alcance): [Wikipedia](https://en.wikipedia.org/wiki/ZALA_Lancet), [Grey Dynamics](https://greydynamics.com/lancet-3-russias-spear-in-the-sky), [GlobalSecurity](https://www.globalsecurity.org/military/world/russia/lancet-specs.htm)
* FPV (velocidade em combate): [Kyiv Post](https://kyivpost.com/post/39057)
* Interceptador Sting (velocidade): [Wild Hornets](https://wildhornets.com/en/sting-interceptor), [Unmanned Airspace](https://www.unmannedairspace.info/?p=34271)
* Ativos: © OpenStreetMap contributors (ODbL), via Overpass API.
