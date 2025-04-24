# agents/classify.py
"""
ClassifyAgent 2.0
─────────────────
• Evalúa la SEVERIDAD de la noticia (−5 … +5) considerando **el texto completo
  y/o el resumen**.
• Usa un paso “razona-luego-responde”: le pedimos al modelo que piense internamente
  (CoT) y después devuelva SOLO el JSON con `score` y `justificacion`.
• Controla que el score sea entero dentro del rango; si el modelo se equivoca,
  lo fuerza a N/D y registra un warning.
"""
from crewai import Agent
from typing import ClassVar
from openai import OpenAI
import json, logging, re

log = logging.getLogger(__name__)

BAROMETRO = """
Clasifica la gravedad con esta escala:

-5: Pérdidas humanas significativas (presencia muertes directas)
-4: Desplazamientos masivos y crisis humanitaria grave (personas desplazadas)
-3: Afectaciones severas en salud y educación
-2: Impacto moderado en servicios básicos
-1: Alteraciones en la vida cotidiana y recursos escasos
 0: Situación controlada pero crítica
 1: Inicio de recuperación o mitigación
 2: Efectos positivos locales
 3: Programas de recuperación eficaces
 4: Recuperación avanzada
 5: Situación positiva y resiliente

INSTRUCCIONES
─────────────
1. Piensa paso a paso (muestra tu razonamiento).
2. Elige el número ENTERO más representativo.
3. Devuelve **exclusivamente** un JSON válido:
   {{"score": INT, "justificacion": "motivo brevemente en ≤25 palabras"}}
"""

JSON_RE = re.compile(r"\{.*\}", re.S)    # para extraer JSON bruto si hace falta

class ClassifyAgent(Agent):
    role: str = "Analista humanitario"
    goal: str = "Asignar puntaje de severidad (–5 … +5)"
    backstory: str = (
        "Especialista en crisis que valora el impacto de desastres naturales "
        "siguiendo un barómetro humanitario estandarizado."
    )

    name: ClassVar[str] = "classifier"
    description: ClassVar[str] = "Clasifica severidad y justifica"

    async def run(self, *, record: dict):
        # 1) Construir prompt — usa texto completo si está, si no el summary.
        body = record.get("text") or record.get("summary") or record["title"]
        body = body[:7000]                      # evita prompts excesivos

        prompt = f"{BAROMETRO}\n\nNoticia:\n\"\"\"{body}\"\"\""

        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        raw_msg = resp.choices[0].message
        payload = raw_msg.json() if hasattr(raw_msg, "json") else raw_msg.content

        # 2) Si el modelo devolvió string, intenta extraer JSON dentro
        if isinstance(payload, str):
            m = JSON_RE.search(payload)
            try:
                payload = json.loads(m.group(0)) if m else json.loads(payload)
            except Exception:
                log.warning("JSON inválido recibido: %s", payload[:120])
                payload = {}

        # 3) Validar score
        score = payload.get("score")
        if not isinstance(score, int) or score < -5 or score > 5:
            log.warning("Score fuera de rango/ausente: %s", score)
            score = "N/D"

        justific = payload.get("justificacion", "N/D")

        record.update({"score": score, "justificacion": justific})
        return record
