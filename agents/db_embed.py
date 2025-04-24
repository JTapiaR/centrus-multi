# agents/db_embed.py
from crewai import Agent
from typing import ClassVar
from openai import OpenAI
import faiss, numpy as np, pandas as pd

client = OpenAI()

class DBEmbedAgent(Agent):
    role: str = "Generador de embeddings de BD"
    goal: str = "Crear/actualizar índice FAISS de tablas tabulares"
    backstory: str = "Convierte filas de CSV/Excel en vectores semánticos."

    name: ClassVar[str] = "db_embedder"
    description: ClassVar[str] = "Embeddings para bases tabulares"

    async def run(self, *, df: pd.DataFrame, faiss_index=None, faiss_payloads=None):
        texts = df.astype(str).agg(" | ".join, axis=1).tolist()
        vecs = [client.embeddings.create(
            model="text-embedding-3-small",
            input=t).data[0].embedding for t in texts]

        dim = len(vecs[0])
        if faiss_index is None:
            faiss_index = faiss.IndexFlatL2(dim)
            faiss_payloads = []
        faiss_index.add(np.array(vecs, dtype="float32"))
        faiss_payloads = np.concatenate([faiss_payloads, np.array(texts)])

        return {"faiss_index": faiss_index, "faiss_payloads": faiss_payloads}
