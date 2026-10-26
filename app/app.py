import os
import sys
from pathlib import Path

# make `from src import ...` work no matter where Streamlit launches from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src import config
from src.retrieval import Retriever
from src.pipeline import answer_question

st.set_page_config(page_title="Ask the PLFS", page_icon="📊", layout="wide")

# expose the Streamlit secret to generation.py, which reads os.environ
try:
    if "GeminiAPIkey" in st.secrets:
        os.environ["GeminiAPIkey"] = st.secrets["GeminiAPIkey"]
except Exception:
    pass  # no secrets file locally: fall back to an environment variable

@st.cache_resource(show_spinner="Loading index and models (about a minute on first run)...")
def get_retriever():
    return Retriever()

st.title("Ask the PLFS")
st.caption("Answers come only from the PLFS Annual Report 2025, with page citations.")

with st.sidebar:
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"])
    k = st.slider("Passages retrieved (k)", 3, 12, config.TOP_K)

st.session_state.setdefault("n_asked", 0)

with st.form("ask"):
    question = st.text_input(
        "Your question",
        placeholder="What was the female labour force participation rate in 2025?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    if st.session_state.n_asked >= 20:
        st.warning("Session limit reached. Refresh the page to continue.")
        st.stop()
    st.session_state.n_asked += 1
    retriever = get_retriever()
    with st.spinner("Searching and answering..."):
        result = answer_question(retriever, question.strip(), k=k, mode=mode)
    st.subheader("Answer")
    st.markdown(result["answer"])
    st.subheader("Sources")
    for n, p in enumerate(result["passages"], 1):
        label = f"[{n}] PDF page {p['page']}"
        if p.get("printed_page"):
            label += f" (printed page {p['printed_page']})"
        with st.expander(label):
            st.text(p["text"])