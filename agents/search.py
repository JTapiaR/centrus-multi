# agents/search.py
import httpx, os, logging, datetime as dt
from typing import ClassVar
from crewai import Agent

log = logging.getLogger(__name__)
NEWS_API = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

class SearchAgent(Agent):
    role: str = "Buscador de noticias"
    goal: str = "Encontrar artículos sobre desastres naturales en México"
    backstory: str = (
        "Eres un investigador experto en fuentes periodísticas y sabes "
        "cómo usar NewsAPI para recuperar información fiable."
    )
    name: ClassVar[str] = "searcher"
    description: ClassVar[str] = "Busca artículos sobre desastres naturales en México"

    async def run(self, *, keywords: str, n: int = 5,
                  date_from: dt.date | None = None,
                  date_to: dt.date   | None = None):
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": keywords,
            "pageSize": n,
            "language": "es",
            "apiKey": NEWS_API
        }
        if date_from: params["from"] = str(date_from)
        if date_to:   params["to"]   = str(date_to)

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                res = await client.get(url, params=params)
                res.raise_for_status()
                data = res.json()["articles"]
            except httpx.HTTPStatusError as e:
                log.warning("NewsAPI falló (%s) - se omitirá: %s", e.response.status_code, e)
                return []          # ← NO lanza excepción; sigue el pipeline
            except Exception as e:
                log.warning("Error en NewsAPI: %s", e)
                return []

        return [
            {
                "title": a["title"],
                "url": a["url"],
                "date": a["publishedAt"],
                "source": "NewsAPI"
            }
            for a in data
        ][:n]
