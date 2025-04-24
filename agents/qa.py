"""
QAAgent
───────
• Busca k=5 vecinos en FAISS.
• Envía el contexto y la pregunta a GPT‑4o.
• Devuelve respuesta breve en español.
"""
from crewai import Agent
from typing import ClassVar
from openai import OpenAI
import numpy as np

client = OpenAI()

QA_PROMPT = """
Contexto:
{context}

Pregunta: {q}

Responde en español con un máximo de 80 palabras:
"""

class QAAgent(Agent):
    role: str = "Asistente de preguntas"
    goal: str = "Responder con información verificada del índice RAG"
    backstory: str = (
        "Utilizas recuperación basada en vectores para contestar preguntas "
        "sobre desastres naturales en México."
    )

    name: ClassVar[str] = "qa"
    description: ClassVar[str] = "Responde preguntas usando RAG"

    async def run(self, *, question: str,
                  faiss_index, faiss_payloads):
        qvec = client.embeddings.create(
            model="text-embedding-3-small",
            input=question).data[0].embedding
        D, I = faiss_index.search(
            np.array([qvec], dtype="float32"), k=5)
        context = "\n\n".join(faiss_payloads[i] for i in I[0])

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": QA_PROMPT.format(context=context, q=question)}])
        return resp.choices[0].message.content.strip()
