# agents/extract.py
"""
Extrae campos estructurados (JSON) de un artículo y verifica su coherencia
mediante cadena de pensamiento breve (CoT).  Devuelve el registro original
enriquecido con un diccionario `data`.
"""
from crewai import Agent
from typing import ClassVar
from openai import OpenAI
import json, logging

log = logging.getLogger(__name__)

# ────────────────────────────
# Plantilla de extracción
# ────────────────────────────
TEMPLATE = """
Extrae la información solicitada del texto a continuación.  Razona internamente
(Cadena de Pensamiento) para asegurarte de que los campos sean coherentes; luego
MUESTRA SOLO el JSON final.

### Formato JSON
{{
  "fecha": "YYYY-MM-DD"                 # fecha del desastre (o descubrimiento)
  "lugar": "País, Estado, Municipio, Región",   # ubicación principal
  "tipo_desastre": "inundación | sismo | huracán | incendio forestal | ...",
  "afectados": "Número aproximado de personas afectadas",
  "muertes_confirmadas": "Número o '0' si no hay",
  "fuente_verificada": true | false     # ¿la fuente menciona confirmación oficial?
}}

### Texto
\"\"\"{texto}\"\"\"
"""

class ExtractAgent(Agent):
    # ─── metadatos requeridos ───
    role: str = "Especialista en extracción de datos"
    goal: str = "Convertir artículos en registros JSON fiables"
    backstory: str = (
        "Eres un analista con gran experiencia identificando información clave "
        "sobre desastres en noticias y reportes oficiales."
    )

    # ─── auxiliares ───
    name: ClassVar[str] = "extractor"
    description: ClassVar[str] = "Extrae campos estructurados y verifica coherencia"

    # ─── ejecución ───
    async def run(self, *, article: dict):
        """
        Parameters
        ----------
        article : dict
            Debe contener 'text' con el cuerpo completo o 'summary' como fallback.
        Returns
        -------
        dict
            El mismo artículo + {"data": {...}}
        """
        body = article.get("text") or article.get("summary", "")
        body = body[:6000]   # límite de tokens razonable

        prompt = TEMPLATE.format(texto=body)
        client = OpenAI()    # api_key via variable de entorno

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        try:
            extracted = resp.choices[0].message.json()        # objeto dict
        except AttributeError:                                # vino como str
            raw = resp.choices[0].message.content
            try:
                extracted = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("ExtractAgent: JSON inválido → %s", raw[:80])
                extracted = {}

        article["data"] = extracted           # SIEMPRE dict, aunque sea vacío
        return article
