# agents/websearch.py  –  versión sin guardar imágenes
from crewai import Agent
import os
from typing import ClassVar
import httpx, feedparser, urllib.parse, asyncio, datetime as dt, re, logging, io
from bs4 import BeautifulSoup            # para fallback rápido
from playwright.sync_api import sync_playwright
import pytesseract
from PIL import Image
from newspaper import Article
from tenacity import retry, wait_exponential, stop_after_attempt

log = logging.getLogger(__name__)


GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
if not GNEWS_API_KEY :
    raise RuntimeError("Configura GNEWS_API_KEY en .env o st.secrets")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
def _download_text(url: str) -> str:
    art = Article(url, language="es")
    art.download(); art.parse()
    return art.text

async def fetch_article(url: str) -> str:
    return await asyncio.to_thread(_download_text, url)

class WebSearchAgent(Agent):
    role: str = "Rastreador GNews"
    goal: str = "Obtener artículos y extraer texto completo"
    backstory: str = (
        "Consulta la API de GNews para encontrar artículos actuales y descarga "
        "el cuerpo con newspaper3k."
    )

    name: ClassVar[str] = "web_searcher"
    description: ClassVar[str] = "GNews.io REST search"

    async def run(
        self,
        *,
        keywords: str,
        n: int = 10,
        date_from: dt.date | None = None,
        date_to: dt.date   | None = None,
    ):
        api = "https://gnews.io/api/v4/search"
        params = {
            "q":        f'{keywords} "México"',   # enfoca a MX
            "lang":     "es",
            "country":  "mx",
            "max":      n,
            "apikey":   GNEWS_API_KEY,
        }
        # rango de fechas (formato YYYY-MM-DD)
        if date_from: params["from"] = date_from.isoformat()
        if date_to:   params["to"]   = date_to.isoformat()

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(api, params=params)
            r.raise_for_status()
            items = r.json().get("articles", [])

        results = []
        for art in items:
            try:
                text = await fetch_article(art["url"])
            except Exception as e:
                log.warning("No se pudo extraer %s: %s", art["url"], e)
                text = ""

            results.append({
                "title":  art["title"],
                "url":    art["url"],
                "date":   art["publishedAt"][:10],  # 2025-04-24T12:00:00Z -> 2025-04-24
                "source": art["source"]["name"],
                #"text":   text,
            })
            if len(results) >= n:
                break
        return results