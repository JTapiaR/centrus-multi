"""
GeoAgent
────────
• Geocodifica el campo `lugar` de cada registro (usa Nominatim‑OSM).
• Añade lat/lon al registro.
• Produce un mapa Folium y devuelve su ruta temporal.
"""
# agents/geo.py
"""
GeoAgent
────────
• Geocodifica el campo `lugar` (si existe) de cada registro usando Nominatim‑OSM.
• Si `lugar` falta, intenta extraer la primera entidad LOC del resumen con spaCy.
• Añade lat/lon al registro.
• Genera un mapa Folium y devuelve su ruta temporal.
El agente es *defensivo*: ignora cualquier elemento que no sea dict o cuyo
campo `data` no sea un diccionario.
"""
from crewai import Agent
from typing import ClassVar
import spacy, httpx, folium, asyncio, tempfile, logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── carga única de spaCy (modelo mediano en español) ──
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    nlp = spacy.blank("es")

class GeoAgent(Agent):
    # ── metadatos obligatorios ──
    role: str = "Geógrafo humanitario"
    goal: str = "Asignar coordenadas y generar un mapa interactivo"
    backstory: str = "Usas Nominatim‑OSM y spaCy para localizar noticias."

    # ── metadatos opcionales (no validados) ──
    name: ClassVar[str] = "geo"
    description: ClassVar[str] = "Geocodifica lugares y crea mapa Folium"

    # ── función auxiliar asíncrona de geocodificación ──
    async def _geocode(self, place: str):
        """Devuelve (lat, lon) o (None, None) si falla."""
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                if r.json():
                    j = r.json()[0]
                    return float(j["lat"]), float(j["lon"])
            except Exception as e:
                log.warning("Nominatim error para '%s': %s", place, e)
        return None, None

    # ── método principal ──
    async def run(self, *, records):
        """records debe ser list[dict] o dict único; devuelve {"records", "map_path"}."""
        # normaliza a lista de dicts
        if isinstance(records, dict):
            records = [records]
        elif not isinstance(records, list):
            records = list(records)

        tasks, valid = [], []

        for rec in records:
            if not isinstance(rec, dict):
                log.warning("GeoAgent ignoró un item no‑dict: %r", rec)
                continue

            # asegura que rec["data"] sea dict (puede venir como str)
            data = rec.get("data", {})
            if not isinstance(data, dict):
                log.warning("GeoAgent: 'data' no‑dict en registro '%s'", rec.get("title"))
                data = {}

            # 1) ¿hay lugar en los datos extraídos?
            partes = [data.get(k, "") for k in ("ciudad","municipio","estado","pais")]
            lugar = ", ".join(p for p in partes if p).strip() or data.get("region","")

            #lugar = data.get("lugar", "").strip()

            # 2) si no, intenta NER sobre summary
            if not lugar:
                doc = nlp(rec.get("summary", ""))
                locs = [e.text for e in doc.ents if e.label_ == "LOC"]
                lugar = locs[0] if locs else ""

            # 3) agenda geocodificación (si hay lugar)
            valid.append(rec)
            tasks.append(
                self._geocode(lugar) if lugar
                else asyncio.sleep(0, result=(None, None))
            )

        # geocodificar en paralelo
        coords = await asyncio.gather(*tasks)

        # construir mapa Folium
        fmap = folium.Map(location=[23, -102], zoom_start=5, tiles="OpenStreetMap")
        for rec, (lat, lon) in zip(valid, coords):
            rec["lat"], rec["lon"] = lat, lon
            if lat is not None:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=f"{rec.get('title','')} (score {rec.get('score','?')})"
                ).add_to(fmap)

        tmp_path = Path(tempfile.gettempdir()) / "map.html"
        fmap.save(tmp_path)

        return {"records": records, "map_path": str(tmp_path)}
