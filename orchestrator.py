# orchestrator.py
"""
Coordina los 7 agentes y va informando el progreso mediante
una función callback que recibe strings.
"""
from agents.search     import SearchAgent
from agents.summarize  import SummarizerAgent
from agents.extract    import ExtractAgent
from agents.classify   import ClassifyAgent
from agents.geo        import GeoAgent
from agents.rag        import RAGAgent
from agents.qa         import QAAgent
from agents.websearch import WebSearchAgent
from agents.db_embed import DBEmbedAgent
from agents.rag import RAGAgent


import asyncio

# instancia única (puedes también instanciar cada vez)
_search  = SearchAgent()
_sum     = SummarizerAgent()
_extract = ExtractAgent()
_class   = ClassifyAgent()
_geo     = GeoAgent()
_rag     = RAGAgent()
_qa      = QAAgent()
_websearch = WebSearchAgent()
_db  = DBEmbedAgent()
qa_agent = _qa
rag_builder = RAGAgent()
qa_agent    = QAAgent()

async def pipeline(keywords: str, n: int,
                   date_from=None, date_to=None,
                   question: str | None = None,
                   progress_cb=lambda msg: None):
    """Devuelve (records, map_path, answer)."""
    progress_cb("🔍 Paso 1· Buscando artículos (NewsAPI + GNews)…")
    raw_news, raw_gnews = await asyncio.gather(
    _search.run(keywords=keywords, n=n,
                date_from=date_from, date_to=date_to),
    _websearch.run(keywords=keywords, n=n,
                   date_from=date_from, date_to=date_to)
)

     # fusionar y deduplicar por URL
    seen = set()
    raw = []
    for item in raw_news + raw_gnews:
       if item["url"] not in seen:
          raw.append(item)
          seen.add(item["url"])

    progress_cb("📝 Paso 2 · Resumiendo, extrayendo y clasificando…")
    processed = []
    for art in raw:
        s  = await _sum.run(article=art)
        e  = await _extract.run(article=s)
        c  = await _class.run(record=e)
        processed.append(c)

    progress_cb("🗺️ Paso 3  · Geocodificando y generando mapa…")
    geo_out = await _geo.run(records=processed)

    progress_cb("📚 Paso 4 · Creando embeddings y RAG…")
    rag_out = await _rag.run(records=geo_out["records"])

    answer = None
    if question:
        progress_cb("💬 Paso 5  · Contestando tu pregunta…")
        answer = await _qa.run(question=question,
                               faiss_index=rag_out["faiss_index"],
                               faiss_payloads=rag_out["faiss_payloads"])
    else:
        progress_cb("✅ Pipeline terminado.")
    return geo_out["records"], geo_out["map_path"], answer

async def add_database(df, rag):
    return await _db.run(df=df, faiss_index=rag["faiss_index"],
                              faiss_payloads=rag["faiss_payloads"])
