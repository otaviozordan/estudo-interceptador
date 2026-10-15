"""Baixa do OpenStreetMap os ativos de infraestrutura de cada local (zonas prioritarias e alvos das missoes
realistas) e grava em data/osm/<local>.json, que e versionado: as missoes ficam reprodutiveis mesmo que o OSM mude.

    python scripts/missoes/osm_ativos.py                       # os 6 locais das missoes realistas
    python scripts/missoes/osm_ativos.py --places Taubate --refazer

Categorias (tag OSM -> categoria, importancia 1..4 usada para escolher ativos defendidos e alvos):
    power=plant (usina) 4, aeroway=aerodrome (aerodromo) 4, power=substation (subestacao) 3,
    man_made=water_works / wastewater_plant (agua) 3, comunicacao (torre/mastro de telecom) 2,
    landuse=industrial com >= 5 ha (industria) 2.
Fonte: Overpass API (varios servidores, com novas tentativas); se todos falharem, Nominatim (so feicoes com nome,
cobertura menor; o arquivo registra a fonte).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from swarm_sim.mission import PLACES, place_origin, slug          # noqa: E402

OUT = os.path.join(ROOT, "data", "osm")
UA = "swarm_intercept-research/1.0 (pesquisa academica; contato otaviozordan@gmail.com)"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
DEFAULT_PLACES = ["Sao Jose dos Campos", "Taubate", "Itajuba (centro)", "Sao Paulo", "Rio de Janeiro", "Brasilia"]
MIN_INDUSTRIAL_HA = 5.0
M_LAT = 111_320.0


def bbox(origin: dict, margin_km: float = 0.5):
    half = origin["size_km"] / 2 + margin_km
    dlat = half * 1000 / M_LAT
    dlon = half * 1000 / (M_LAT * math.cos(math.radians(origin["lat"])))
    return origin["lat"] - dlat, origin["lon"] - dlon, origin["lat"] + dlat, origin["lon"] + dlon


def classify(tags: dict) -> tuple[str, int] | None:
    if tags.get("power") == "plant":
        return "usina", 4
    if tags.get("aeroway") == "aerodrome":
        return "aerodromo", 4
    if tags.get("power") == "substation":
        return "subestacao", 3
    if tags.get("man_made") in ("water_works", "wastewater_plant"):
        return "agua", 3
    if tags.get("man_made") in ("communications_tower", "mast", "tower") and (
            tags.get("tower:type") == "communication" or tags.get("man_made") == "communications_tower"):
        return "comunicacao", 2
    if tags.get("landuse") == "industrial":
        return "industria", 2
    return None


def _post(url: str, query: str, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def overpass(bb, rounds: int = 4, log=print) -> list[dict] | None:
    s, w, n, e = bb
    b = f"({s:.5f},{w:.5f},{n:.5f},{e:.5f})"
    q = (f'[out:json][timeout:150];('
         f'nwr["power"~"^(substation|plant)$"]{b};nwr["aeroway"="aerodrome"]{b};nwr["landuse"="industrial"]{b};'
         f'nwr["man_made"~"^(water_works|wastewater_plant|communications_tower)$"]{b};'
         f'nwr["tower:type"="communication"]{b};);out center bb tags;')
    for k in range(rounds):
        for url in OVERPASS:
            try:
                return _post(url, q)["elements"]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as ex:
                log(f"    {url.split('/')[2]}: {getattr(ex, 'code', '')} {type(ex).__name__}")
        time.sleep(20 * (k + 1))
    return None


def nominatim(bb, log=print) -> list[dict]:
    """Fallback: busca por categoria (so feicoes com nome)."""
    s, w, n, e = bb
    out = []
    for q in ("[power=substation]", "[power=plant]", "[aeroway=aerodrome]", "[landuse=industrial]",
              "[man_made=water_works]", "[man_made=wastewater_plant]", "[man_made=communications_tower]"):
        url = ("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
            {"q": q, "viewbox": f"{w},{n},{e},{s}", "bounded": 1, "format": "jsonv2", "limit": 50,
             "extratags": 1}))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                for x in json.load(r):
                    bbx = [float(v) for v in x.get("boundingbox", [x["lat"], x["lat"], x["lon"], x["lon"]])]
                    out.append({"type": x.get("osm_type", ""), "id": x.get("osm_id"),
                                "tags": {x["category"]: x["type"], "name": x.get("name", "")},
                                "center": {"lat": float(x["lat"]), "lon": float(x["lon"])},
                                "bounds": {"minlat": bbx[0], "maxlat": bbx[1], "minlon": bbx[2], "maxlon": bbx[3]}})
        except Exception as ex:                                   # noqa: BLE001
            log(f"    nominatim {q}: {ex}")
        time.sleep(1.2)                                           # politica de uso: 1 requisicao/s
    return out


def to_assets(elements: list[dict], origin: dict) -> list[dict]:
    lat0, lon0 = origin["lat"], origin["lon"]
    m_lon = M_LAT * math.cos(math.radians(lat0))
    seen, out = set(), []
    for el in elements:
        tags = el.get("tags", {})
        c = classify(tags)
        if c is None:
            continue
        cat, imp = c
        bd = el.get("bounds")
        ctr = el.get("center") or ({"lat": el["lat"], "lon": el["lon"]} if "lat" in el else None)
        if not ctr and bd:                                        # alguns servidores mandam so o retangulo
            ctr = {"lat": 0.5 * (bd["minlat"] + bd["maxlat"]), "lon": 0.5 * (bd["minlon"] + bd["maxlon"])}
        if not ctr:
            continue
        if bd:
            h = (bd["maxlat"] - bd["minlat"]) * M_LAT
            w = (bd["maxlon"] - bd["minlon"]) * m_lon
            area_ha, radius = h * w / 1e4, 0.5 * math.hypot(h, w)
        else:
            area_ha, radius = 0.0, 0.0
        if cat == "industria" and area_ha < MIN_INDUSTRIAL_HA:
            continue
        if cat == "industria" and area_ha >= 50:
            imp = 3
        key = (el.get("type"), el.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"categoria": cat, "importancia": imp, "nome": tags.get("name", ""),
                    "lat": ctr["lat"], "lon": ctr["lon"],
                    "n": (ctr["lat"] - lat0) * M_LAT, "e": (ctr["lon"] - lon0) * m_lon,
                    "raio_m": round(radius, 1), "area_ha": round(area_ha, 2), "osm": f"{el.get('type')}/{el.get('id')}"})
    return out


def fetch(place: str, refazer: bool = False, log=print) -> dict:
    path = os.path.join(OUT, slug(place.split("(")[0]) + ".json")
    if os.path.exists(path) and not refazer:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    origin = place_origin(place)
    bb = bbox(origin)
    log(f"  {place}: Overpass ...")
    els = overpass(bb, log=log)
    fonte = "overpass"
    if els is None:
        log(f"  {place}: Overpass indisponivel, usando Nominatim (cobertura parcial)")
        els, fonte = nominatim(bb, log), "nominatim"
    d = {"local": place, "origin": origin, "fonte": fonte, "data": date.today().isoformat(),
         "licenca": "(c) OpenStreetMap contributors, ODbL", "ativos": to_assets(els, origin)}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", nargs="+", default=DEFAULT_PLACES)
    ap.add_argument("--refazer", action="store_true")
    a = ap.parse_args()
    for p in a.places:
        assert p in PLACES, p
        d = fetch(p, a.refazer)
        cats: dict = {}
        for x in d["ativos"]:
            cats[x["categoria"]] = cats.get(x["categoria"], 0) + 1
        print(f"{p}: {len(d['ativos'])} ativos ({d['fonte']}) {cats}", flush=True)


if __name__ == "__main__":
    main()
