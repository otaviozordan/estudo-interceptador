# modelos/

Parâmetros treinados dos métodos M23–M29, uma pasta por método (`<id>_<chave>`), lidos por `rodar.py` e pelo benchmark:

| pasta | conteúdo | treino |
|---|---|---|
| `M23_arvore_comportamento/coef.json` | coeficientes c1..c4 da utilidade afim | `python scripts/treinar.py M23` |
| `M24_IL-DS/s<semente>.pt` | imitação, DeepSets | `python scripts/treinar.py M24 --semente 0 1 2` |
| `M25_IL-ATT/s<semente>.pt` | imitação, atenção | `... M25` |
| `M26_IL+PPO-DS/s<semente>.pt` | imitação + PPO, DeepSets (refina o M24 da mesma semente) | `... M26` |
| `M27_IL+PPO-ATT/s<semente>.pt` | imitação + PPO, atenção (refina o M25) | `... M27` |
| `M28_PPO-DS/s<semente>.pt` | PPO do zero, DeepSets | `... M28` |
| `M29_PPO-ATT/s<semente>.pt` | PPO do zero, atenção | `... M29` |

Sem `coef.json`, o M23 usa os coeficientes padrão (não ajustados). Os pesos `.pt` são `state_dict` do PyTorch.
