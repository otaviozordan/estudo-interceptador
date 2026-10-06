# swarm

Projeto de pesquisa para artigo de congresso.

Os dados de terreno (`data/dem/*.tif`, Copernicus GLO-30) não são versionados e ficam só localmente.

## Simulação (`swarm_sim/`)

UAVs de asa fixa 6-DOF (quaternions, sustentação/arrasto/estol, empuxo, ISA, vento + rajada) voando sobre o DEM, com
detecção de colisão com o terreno (e perda de separação entre aeronaves). Referencial local NED (n = norte, e = leste, metros).

| módulo | conteúdo |
|---|---|
| `terrain.py` | DEM GeoTIFF → grade métrica local, altura/AGL/declividade, colisão de segmento, DEM fractal sintético |
| `aircraft.py` | dinâmica 6-DOF (Aerosonde), atuadores, trim |
| `guidance.py` | L1 em segmentos, órbita (loiter), altitude AGL com antecipação de terreno, waypoint no chão |
| `autopilot.py` | rolagem/arfagem/velocidade em cascata |
| `routes.py` | patrulha (hipódromo), cobertura (lawn-mower), inspeção (corredor + órbitas) |
| `mission.py` | arquivo de missão JSON (aeronaves, waypoints, vento, spline, rajadas, radares) |
| `radar.py` | estações de radar (medidas ruidosas, linha de visada no DEM, Pd) + rastreador multi-radar (Kalman) |

### Fluxo com missão
```
python mission_builder.py missions/minha.json [--lat -23.03 --lon -45.56 --size 10]   # construtor (Tkinter)
python run_mission.py missions/minha.json            # simula até todas as aeronaves colidirem/saírem do mapa (limite: duração da missão) e abre o replay
python run_mission.py missions/minha.json --resim    # ignora o cache e simula de novo (o resultado fica em results/cache/ e o replay seguinte é instantâneo)
python run_mission.py missions/minha.json --batch    # sem janela: duração da missão + PNGs em results/
```
**Construtor**: clique esquerdo adiciona (modos Waypoint / Chão (impacto) / Loiter / Início), arrastar move o mapa,
scroll dá zoom, clique direito remove. Alterne **Elevação ↔ Satélite** (imagens Esri baixadas sob demanda para `data/tiles/`,
precisa de internet; ~4 m/px em mapas de 10 km, ideal para planejar sobre cidades) e recentre o mapa por lat/lon/lado.
Waypoint **no chão** (`agl <= 0`, ou `Waypoint.ground(n, e)`) faz a aeronave mergulhar e colidir com o terreno.

**Construtor**: tema escuro com abas (Aeronaves / Rota / Mapa), **arrastar** waypoints e início, grade com escala, rumo/distância por
trecho e **rota suave (spline Catmull-Rom)** que passa por todos os waypoints (na simulação vira waypoints densos a cada ~150 m).
**Rajadas localizadas**: modo ≋ Rajada (tecla 5) ou aba *Rajadas*; clique no mapa (perto de um waypoint, encaixa nele), arraste o centro para mover e a ponta da seta para girar. Cada zona tem velocidade, rumo (para onde sopra), raio e componente vertical; o vento é máximo no centro e cai suavemente a zero na borda, e a aeronave o sente enquanto estiver dentro do círculo (`swarm_sim/gusts.py`, aplicado no 6-DOF; o replay destaca a zona ativa).
Atalhos: 1–5 modos, Ctrl+S, Ctrl+Z, Del. 

**Locais prontos**: `--place itajuba` (ou o seletor "Local"): Itajubá (centro, aeroporto SBIT e região de 40 km), Pouso Alegre,
Campos do Jordão, Taubaté, Pindamonhangaba, Guaratinguetá, São José dos Campos. O DEM local cobre ~ -24..-22 °S, -46..-45 °W
(relevo da Mantiqueira é íngreme: declividade mediana ~16° na região de Itajubá).

**Velocidade por trecho**: o campo *Vel. do trecho* vale para os próximos waypoints (velocidade do ar do trecho até ele); selecione
um waypoint (lista ou clique nele) e use *Aplicar ao selecionado*, ou *Vel. p/ todos*. Faixa 19–38 m/s, com rampa de 2 m/s².

**Replay** (`swarm_sim/viewer.py`): bolinha seguindo a aeronave (opção *Seguir*), rota **teórica** (tracejada, waypoints/órbitas)
sobreposta à trajetória **real** (linha cheia), parâmetros de voo (altitude, AGL, Va, Vz, φ/θ/ψ, α, p/q/r, erro de rota,
waypoint/modo), barras de comandos (profundor, aileron, leme, manete: comando × posição real do atuador) e gráficos
real × comando (altitude, AGL, Va, rolagem). Teclas: espaço play/pausa, ←/→ ±1 s, F seguir, R reiniciar; clique nos gráficos para saltar no tempo.

### Radares (camada de informação)
A posição das aeronaves só é conhecida **pelos radares** (`swarm_sim/radar.py`). No construtor: modo **📡 Radar** (tecla 6) ou aba *Radares*;
clique no mapa para posicionar quantas estações quiser, arraste para mover, edite os parâmetros da selecionada (alcance Pd=50 %, altura da antena,
período de varredura, σ de distância/azimute/elevação, Pd máx., elevação mín./máx. (cone de silêncio), setor de azimute, zona cega, falsos alarmes).
O hachurado mostra a **cobertura** para um alvo a uma AGL escolhida (linha de visada contra o DEM, curvatura da Terra com raio efetivo 4/3).

Modelo (`swarm_sim/radar.py`):
- **Equação do radar**: `SNR = SNR50·(R0/r)⁴·RCS_efetiva·perdas`; `R0` ("alcance para 1 m²") é onde Pd = 50 % em visada livre. Alvo flutuante
  Swerling I: `Pd = Pfa^(1/(1+SNR))`. **RCS por aeronave** (campo na aba Aeronaves; alcance ∝ RCS^¼) com **dependência do aspecto** (0,3× de frente/cauda, 1,7× de lado).
- **Relevo**: difração em gume de faca (zonas de Fresnel, depende da frequência) + curvatura da Terra (raio efetivo 4/3): a sombra atenua gradualmente
  o sinal em vez de liga/desliga. Atenuação atmosférica em dB/km.
- **Varredura real**: o feixe gira (ou varre o setor) e cada alvo é medido **no instante em que o feixe passa por ele**; radares são assíncronos.
- **Ruído**: σ nominal em SNR de referência, escalando com 1/√SNR (Cramér-Rao); **vieses** sistemáticos de distância e azimute por radar (desalinhamento) que o
  rastreador não conhece; falsos alarmes concentrados perto do radar (clutter).
- **Rastreador**: IMM (dois modelos de velocidade constante, aceleração 1 e 8 m/s², para reta/manobra) sobre medidas convertidas para cartesiano, associação por
  Mahalanobis + húngaro, confirmação com 3 plots, descarte por tempo sem atualização (20 s) e fusão de duplicatas. O tracker nunca usa a identidade verdadeira (só a avaliação: erro RMS, % rastreada).
- Não modelado: Doppler/clutter de solo detalhado, multipercurso, abertura do feixe em elevação, latência, JPDA/MHT.
Controladores/interceptadores devem usar apenas `sim.radar_net.picture(t)` (pistas com posição, velocidade, incerteza e idade).
`run_mission.py` imprime por aeronave a % de tempo rastreada, 1ª pista e erro RMS; o replay mostra radares, plots (+), pistas (quadrados) e o gráfico de erro.
No replay, **Visão radar** esconde a verdade (aeronaves, rotas, rajadas, telemetria) e mostra só radares, plots e uma tabela de pistas; o seletor *Parâmetros do radar* exibe a configuração interna do radar escolhido e o estado da última varredura. Relevo íngreme bloqueia muito: com antena a 20 m do solo e alvo a 100 m AGL, radares no vale não veem nada — posicione-os em morros.

Demo antiga com 3 rotas: `python run_demo.py` (se presente). Imagens: © Esri, Maxar, Earthstar Geographics.
