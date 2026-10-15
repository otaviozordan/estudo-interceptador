"""M8 -- Hungaro ponderado SEM o filtro de plausibilidade (E1): mede quanto o filtro vale."""
from __future__ import annotations

from .base import Metodo, politica
from .m07_hungaro_ponderado import hungarian_priority

METODO = Metodo("M08", "hungaro_prio_sem_filtro", "Húngaro pond. sem filtro", "Atribuição ótima",
                lambda _: politica(hungarian_priority, filt=False))
