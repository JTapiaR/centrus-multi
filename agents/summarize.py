# agents/summarize.py
"""
SummarizerAgent
───────────────
1. Descarga el texto completo del artículo con `newspaper3k`.
2. Genera un resumen de 3‑5 frases usando GPT‑4o.
3. Devuelve el artículo enriquecido con campos `text` y `summary`.
"""
from crewai import Agent
from typing import ClassVar
from openai import OpenAI
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

# prompt de resumen
SUM_PROMPT = """
Resume en 3‑5 frases los hechos clave de la noticia siguiente.
No inventes datos.  El resumen debe ser en español formal.

Noticia:
\"\"\"{texto}\"\"\"
"""

class SummarizerAgent(Agent):
    # ─── metadatos requeridos por crewai ───
    role: str = "Analista de contenidos"
    goal: str = "Generar resúmenes fiables y concisos de cada noticia"
    backstory: str = (
        "Eres periodista de investigación con años de experiencia resumiendo "
        "información sobre desastres naturales para organismos humanitarios."
    )

    # opcionales (no validados por Pydantic)
    name: ClassVar[str] = "summarizer"
    description: ClassVar[str] = "Resume artículos"

    # ─── utilidades ───
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def _get_article_text(self, url: str) -> str:
        """Descarga y parsea el artículo, con reintentos exponenciales."""
        art = Article(url, language="es")
        art.download()
        art.parse()
        return art.text

    async def run(self, *, article: dict):
        """
        Parameters
        ----------
        article : dict
            Debe contener al menos 'url' y 'title'.
        Returns
        -------
        dict
            Mismo dict con campos 'text' y 'summary' añadidos.
        """
        # 1) Descargar texto completo en hilo aparte
        try:
            full_text = await asyncio.to_thread(self._get_article_text,
            article["url"])
        except Exception as e:
            self.logger.warning("newspaper3k falló para %s: %s",
                            article["url"], e)
        full_text = article["title"]

        # 2) Llamar al modelo para resumir
        client = OpenAI()                       # api_key vía entorno
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": SUM_PROMPT.format(texto=full_text[:7000])
            }]
        )
        summary = resp.choices[0].message.content.strip()

        # 3) Enriquecer registro y devolver
        article.update({"text": full_text, "summary": summary})
        return article
