# agents/rag.py
"""
RAGAgent 2.0
────────────
1. -- Construye (o amplía) un índice FAISS con embeddings OpenAI.
2. -- Guarda metadatos (title, url, score…) junto al embedding.
3. -- Expone un método `query()` que:
      • Recupera los *k* documentos más próximos  
      • Usa MMR (diversidad) para evitar repeticiones  
      • Redacta la respuesta con historial de conversación
        para mantener un chat largo y coherente.
"""
from crewai import Agent
from typing import ClassVar
from openai import OpenAI
import faiss, numpy as np, json, logging, heapq
from dataclasses import asdict, dataclass 


log = logging.getLogger(__name__)

#──── helpers ────────────────────────────────────────────
def _embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Devuelve el embedding OpenAI del texto."""
    openai_client = OpenAI()                       # ← instancia local
    return (
        openai_client
        .embeddings
        .create(model=model, input=text)
        .data[0]
        .embedding
    )

def mmr(query_vec, doc_vecs, top_k=5, lambda_=0.5):
    """Maximal-Marginal-Relevance sobre vectores numpy."""
    selected, idxs = [], list(range(len(doc_vecs)))
    sims = doc_vecs @ query_vec
    while idxs and len(selected) < top_k:
        if not selected:
            i = int(np.argmax(sims[idxs]))
            selected.append(idxs.pop(i))
            continue
        mmr_scores = []
        for j in idxs:
            diversity = max(np.dot(doc_vecs[j], doc_vecs[s]) for s in selected)
            score = lambda_ * sims[j] - (1 - lambda_) * diversity
            mmr_scores.append(score)
        i = int(np.argmax(mmr_scores))
        selected.append(idxs.pop(i))
    return selected

def _get_summary(item) -> str:
    """Extrae el campo summary sin lanzar excepciones."""
    # 1) dataclass / objeto con atributo
    if hasattr(item, "summary"):
        return getattr(item, "summary", "")
    # 2) diccionario u objeto mapeable
    if isinstance(item, dict):
        return item.get("summary", "")
    # 3) cualquier otro: lo convertimos a str y truncamos
    return str(item)[:200]

#────────────────── payload dataclass ──────────────────#
@dataclass
class Doc:
    id: int
    summary: str
    meta: dict        # título, url, score, etc.

#────────────────── agente ──────────────────#
class RAGAgent(Agent):
    role: str = "Ingeniero de conocimiento"
    goal: str = "Construir y explotar un índice vectorial robusto"
    backstory: str = "Especialista en embeddings y búsqueda semántica."

    name: ClassVar[str] = "rag_builder"
    description: ClassVar[str] = "Índice FAISS + MMR + chat context"

    #───────── build / update index ─────────#
    async def run(self, *, records: list[dict],
                  faiss_index=None, faiss_payloads: list[Doc] | None = None):

        # 1) Embeddings de los resúmenes
        vecs = [_embed(r["summary"]) for r in records]
        dim  = len(vecs[0])

        if faiss_index is None:
            faiss_index = faiss.IndexFlatIP(dim)      # producto interno (== coseno)
            faiss_payloads = []

        start_id = len(faiss_payloads)
        faiss_index.add(np.array(vecs, dtype="float32"))
        faiss_payloads.extend(
            [Doc(id=start_id+i, summary=r["summary"], meta=r)
             for i, r in enumerate(records)]
        )

        return {"faiss_index": faiss_index,
                "faiss_payloads": faiss_payloads,
                "records": records}

    #───────── ask / chat ─────────#
    async def query(self, *,
                    question: str,
                    faiss_index,
                    faiss_payloads: list[Doc],
                    history: list[dict] | None = None,
                    k: int = 5) -> tuple[str, list[dict]]:

        # Embedding de la pregunta
        q_vec = np.array(_embed(question), dtype="float32")[None, :]
        # Similaridad por producto interno
        _, _ = faiss_index.search(q_vec, len(faiss_payloads))  # necesitamos doc_vecs
        doc_vecs = faiss_index.reconstruct_n(0, len(faiss_payloads))

        # MMR para variedad
        best = mmr(q_vec[0], doc_vecs, top_k=k)

        context = "\n\n".join(
            f"[{i+1}] {_get_summary(faiss_payloads[i])}" for i in best
        )

        sys_prompt = ("Eres un asistente experto en desastres naturales. "
                      "Responde SOLO con la información del CONTEXTO. "
                      "Si la respuesta no está presente, contesta 'No encontrado'.")
        user_prompt = f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"
        chat_history = history or []

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(chat_history[-4:])            # máx 4 turnos previos
        messages.append({"role": "user", "content": user_prompt})

        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        answer = resp.choices[0].message.content.strip()

        # agrega turnos al historial
        chat_history = chat_history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer}
        ]

        return answer, chat_history