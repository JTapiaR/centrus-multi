# app.py
# ────────────────────────────────────────────────────────────────
# Streamlit UI para el sistema multi-agente:
#   • Búsqueda de noticias
#   • Ejecución de agentes (7 pasos)
#   • Mapa + CSV
#   • RAG combinado con BD externa
# ────────────────────────────────────────────────────────────────
import os, asyncio, datetime as dt, urllib.parse, requests, pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from orchestrator import pipeline, add_database, qa_agent, rag_builder
# ───────── función auxiliar ─────────
def mask_key(key: str | None, show: int = 4) -> str:
    """
    Devuelve la clave con solo los primeros y últimos `show` caracteres visibles.
    Ej.:  sk-ab…YZSN  ->  sk-ab••••••YZSN
    """
    if not key:
        return "⨉ (no definida)"
    if len(key) <= show * 2:
        return key[0] + "•" * (len(key) - 2) + key[-1]
    return key[:show] + "•" * (len(key) - show * 2) + key[-show:]

st.set_page_config(page_title="📰 Búsqueda y Resumen de Noticias sobre Desastres Naturales en México",
                   page_icon="📰", layout="wide")
#──────────────────── UI global ─────────────────────────#
lang = st.sidebar.selectbox("Language / Idioma", ("Español", "English"))
st.sidebar.image("logocentrus.png")
with st.sidebar.expander("🔑 Ver claves cargadas"):
    openai_key = os.getenv("OPENAI_API_KEY")
    news_key   = os.getenv("NEWS_API_KEY")
    gnews_key  = os.getenv("GNEWS_API_KEY")

    st.code(f"OPENAI_API_KEY = {mask_key(openai_key)}", language="text")
    st.code(f"NEWS_API_KEY   = {mask_key(news_key)}",   language="text")
    st.code(f"GNEWS_API_KEY  = {mask_key(gnews_key)}",  language="text")

st.divider()

# cargar variables .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
#GNEWS_API_KEY = "d579856ef730eac4de9248ccb81a002f"
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# importa orquestador y helper para BD
from orchestrator import pipeline, add_database, qa_agent, rag_builder

#──────────────────── textos ES/EN ────────────────────#
TX_ES = {
    "app_title":   "📰 Búsqueda y Resumen de Noticias sobre desastres naturales en México",
    "app_desc":    "Esta aplicación te permite buscar artículos de noticias recientes y extraer automáticamente información clave sobre desastres naturales en México utilizando un sistema multi‑agente basado en GPT‑4o.",
    "expl":        """
### Cómo Funciona la Aplicación
1. **Ingresa Palabras Clave y Número de Resultados**: Proporciona las palabras clave (p.ej. 'inundación', 'sismo', etc.) y el número de artículos recientes que deseas recuperar.
2. **Buscar Noticias**: Haz clic para obtener artículos reales.
3. **Selecciona Artículos**: Elige cuáles analizar.
4. **Procesar**: Se disparan 7 agentes (búsqueda, resumen, extracción, clasificación, geocodificación, embeddings y QA) y verás su progreso en tiempo real.
5. **Resultados**: Tabla, mapa interactivo, descarga CSV y respuesta a tu pregunta opcional.
""",
    "step1": "1️⃣ Palabras clave y número de resultados",
    "step1_desc": "Escribe términos de búsqueda y cuántos artículos recuperar.",
    "step2": "2️⃣ Buscar noticias",
    "step2_desc": "Pulsa para consultar NewsAPI.",
    "step3": "3️⃣ Seleccionar artículos",
    "step3_desc": "Marca los artículos que quieres procesar.",
    "step4_title": "4️⃣ Ejecutar agentes",
    "step4_desc": "Los agentes resumirán, extraerán y mapearán la información.",
    "keywords":  "Palabras clave:",
    "from":      "Desde",
    "to":        "Hasta",
    "num":       "Número de artículos:",
    "search_bt": "Buscar Noticias",
    "warn_kw":   "Ingresa palabras clave válidas.",
    "no_art":    "No se encontraron artículos.",
    "api_err":   "Error en la API de noticias.",
    "sel_instr": "Selecciona los artículos a analizar:",
    "process_bt":"Procesar con multi-agente",
    "no_sel":    "Selecciona al menos un artículo.",
    "results":   "Resultados",
    "download":  "Descargar CSV",
    "db_upload": "📂 Subir base de datos adicional (CSV/XLSX)",
    "add_idx":   "Agregar al índice",
    "ask":       "❓ Preguntar al asistente",
    "rerun":     "🔄 Nueva búsqueda",
}
TX_EN = {
    **{k:v for k,v in TX_ES.items()},    # duplica claves
    "app_title": "📰 Natural-Disaster News · Search & RAG (Mexico)",
    "app_desc":  "Multi-agent pipeline to search news, extract data and build a hybrid RAG index.",
    "keywords":  "Keywords:",
    "from":      "From",
    "to":        "To",
    "num":       "Number of articles:",
    "search_bt": "Search News",
    "warn_kw":   "Please enter valid keywords.",
    "no_art":    "No articles found.",
    "api_err":   "News API error.",
    "sel_instr": "Select articles:",
    "process_bt":"Process with multi-agent",
    "no_sel":    "Select at least one article.",
    "download":  "Download CSV",
    "db_upload": "📂 Upload additional database (CSV/XLSX)",
    "add_idx":   "Add to index",
    "ask":       "❓ Ask the assistant",
    "rerun":     "🔄 New search",
}



#──────────────────── Paso 1 ────────────────────────────#
T = TX_ES if lang == "Español" else TX_EN
# ╭─ Encabezado principal
st.title(TX_ES["app_title"])
st.markdown(f"##### {T['app_desc']}")
st.markdown(T["expl"], unsafe_allow_html=True)
st.divider()

st.subheader(T["step1"])
col1, col2, col3 = st.columns([2,1,1])
with col1:
    keywords = st.text_input(T["keywords"])
with col2:
    date_from = st.date_input(T["from"], value=dt.date.today()-dt.timedelta(days=90))
with col3:
    date_to   = st.date_input(T["to"],   value=dt.date.today())
num_results = st.number_input(T["num"], 1, 30, 5)

#──────────────────── Paso 2 ────────────────────────────#
st.subheader(T["step2"])
if st.button(T["search_bt"]):
    if not keywords.strip():
        st.warning(T["warn_kw"])
    else:
        q  = urllib.parse.quote_plus(keywords)
        url = (f"https://newsapi.org/v2/everything?q={q}&pageSize={num_results}"
               f"&language=es&apiKey={NEWS_API_KEY}")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                arts = r.json().get("articles", [])
                st.session_state.articles = [
                    {"Título": a["title"], "Fecha": a["publishedAt"], "Enlace": a["url"], "Fuente":"NewsAPI"}
                    for a in arts
                ]
                if not arts:
                    st.info(T["no_art"])
            else:
                st.error(T["api_err"])
        except Exception as e:
            st.error(f"{T['api_err']}: {e}")

#──────────────────── Paso 3 ────────────────────────────#
st.subheader(T["step3"])
sel = []
if st.session_state.get("articles"):
    df = pd.DataFrame(st.session_state.articles)
    st.dataframe(df, use_container_width=True)
    sel = st.multiselect(T["sel_instr"], df.index, format_func=lambda i: df.loc[i,"Título"])

#──────────────────── Paso 4 ────────────────────────────#
st.subheader(T["step4_title"])
st.write(T["step4_desc"])
if st.button(T["process_bt"], disabled=not sel):
    if not sel:
        st.warning(T["no_sel"])
    else:
        arts = [dict(title=df.loc[i,"Título"], url=df.loc[i,"Enlace"],
                     date=df.loc[i,"Fecha"], source=df.loc[i,"Fuente"])
                for i in sel]
        with st.spinner("⏳ Ejecutando agentes…"):
            records, map_path, rag = asyncio.run(
                pipeline(keywords=keywords, n=len(arts),
                         date_from=date_from, date_to=date_to,
                         question=None,
                         progress_cb=lambda m: st.write(m))
            )
            st.session_state.results_df = pd.json_normalize(records, sep="_")
            st.session_state.map_path   = map_path
            st.session_state.rag        = rag   # diccionario con índice FAISS
        st.success("Agentes finalizados.")

#──────────────────── Mostrar resultados ────────────────#
if st.session_state.get("results_df") is not None:
    st.subheader(T["results"])
    st.dataframe(st.session_state.results_df, use_container_width=True)
    html_map = Path(st.session_state.map_path).read_text(encoding="utf-8")
    st.components.v1.html(html_map, height=500, scrolling=False)
    st.download_button(T["download"],
                       st.session_state.results_df.to_csv(index=False, encoding="utf-8"),
                       file_name="resultados.csv", mime="text/csv")

# ───────── subir BD externa ─────────
st.divider()
st.subheader(T["db_upload"])
up = st.file_uploader("", type=["csv", "xlsx"])
if up:
    new_df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.write("Filas detectadas:", len(new_df))

    if st.button(T["add_idx"]):
        with st.spinner("Embebiendo y fusionando…"):
            base_rag = st.session_state.get("rag") or {"faiss_index": None,
                                                       "faiss_payloads": []}
            st.session_state.rag = asyncio.run(
                add_database(new_df, base_rag)
            )
        st.success("Base añadida al índice.")

# ───────── Preguntas RAG ─────────
# ───────── Preguntar al asistente (RAG) ─────────
st.subheader(T["ask"])

q = st.text_input("Pregunta:")

# El botón solo se habilita si hay texto y ya existe un índice RAG
resp_disabled = not (q.strip() and st.session_state.get("rag"))

iresp_disabled = not (q.strip() and st.session_state.get("rag"))

if st.button("Responder", disabled=resp_disabled):
    answer, st.session_state.chat = asyncio.run(
        rag_builder.query(
            question        = q,
            faiss_index     = st.session_state.rag["faiss_index"],
            faiss_payloads  = st.session_state.rag["faiss_payloads"],
            history         = st.session_state.get("chat", [])
        )
    )
    st.write(answer)

