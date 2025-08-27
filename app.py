# app.py
import os
import tempfile
import json
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# import your existing pipeline helpers
from rag_chain import ingest_pdf, slug_namespace, answer_with_citations

load_dotenv()

st.set_page_config(page_title="📚 RAG Book Chatbot", page_icon="📖", layout="wide")

# ---------- CSS / look & feel ----------
CSS = """
<style>
/* page container */
.block-container{padding-top:1.5rem;padding-bottom:2rem}
/* chat bubbles */
.chat-row{display:flex; gap:12px; margin-bottom:10px;}
.chat-user{margin-left:auto; max-width:75%; background:linear-gradient(90deg,#e6f0ff,#dfefff); padding:12px 14px; border-radius:14px; border:1px solid #cddffb;}
.chat-bot{margin-right:auto; max-width:75%; background:#ffffff; padding:12px 14px; border-radius:14px; border:1px solid #eee;}
.meta {font-size:0.82rem; color:#666; margin-top:6px;}
.small {font-size:0.85rem; color:#444;}
.sidebar-section {padding:10px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Sidebar: ingestion + settings ----------
with st.sidebar:
    st.title("📚 Book Indexing")
    st.markdown("Upload a **200+ page** PDF (or use the CLI `ingest.py`).")
    uploaded = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False)

    default_ns = "my-book"
    if uploaded:
        default_ns = slug_namespace(os.path.splitext(uploaded.name)[0])

    namespace_input = st.text_input("Namespace (book id)", value=default_ns,
                                    help="A short name for the book. Used as Pinecone namespace.")
    index_button = st.button("📥 Index / Re-index Book")

    st.markdown("---")
    st.markdown("**Chat Settings**")
    k_retrieval = st.slider("Top k retrieved chunks (UI only — rag_chain uses k=6 default)", 1, 10, 6)
    show_sources_by_default = st.checkbox("Auto-expand sources after each answer", value=False)
    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Show Last Indexed Namespace"):
        st.info(f"Namespace: `{namespace_input}`")

# ---------- Main layout ----------
left_col, right_col = st.columns([3, 1])

# session state
if "messages" not in st.session_state:
    # messages = list of (role, text, meta)
    st.session_state.messages = []
if "namespace" not in st.session_state:
    st.session_state.namespace = namespace_input

# handle indexing
if index_button:
    if not uploaded:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        # save uploaded pdf to temp file and call ingest_pdf
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name
        ns = slug_namespace(namespace_input)
        st.sidebar.info(f"Indexing into namespace: `{ns}` — this can take several minutes for large books.")
        try:
            with st.spinner("Chunking, embedding and upserting to Pinecone..."):
                stats = ingest_pdf(tmp_path, ns)
            st.sidebar.success(f"Ingested {stats['chunks']} chunks from {stats['pages']} pages into '{ns}'.")
            st.session_state.namespace = ns
            # add a system message to history
            st.session_state.messages.append(("system", f"Indexed book → {ns} (chunks: {stats['chunks']})", {"time": datetime.utcnow().isoformat()}))
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# render chat history
with left_col:
    st.markdown("## 📖 Chat")
    for role, text, meta in st.session_state.messages:
        if role == "user":
            st.markdown(f"<div class='chat-row'><div class='chat-user'>{st.session_state.get('user_prefix','You')}: {text}</div></div>", unsafe_allow_html=True)
        elif role == "assistant":
            # include small meta if present
            meta_text = ""
            if meta and meta.get("source_count"):
                meta_text = f"<div class='meta'>Sources: {meta['source_count']}</div>"
            st.markdown(f"<div class='chat-row'><div class='chat-bot'><b>Assistant:</b> {text}{meta_text}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-row'><div class='chat-bot small'>{text}</div></div>", unsafe_allow_html=True)

    # input area
    query = st.text_input("Ask a question about the indexed book", key="query_input")
    if st.button("Ask") and query:
        ns = st.session_state.get("namespace") or namespace_input
        st.session_state.messages.append(("user", query, {"time": datetime.utcnow().isoformat()}))
        with st.spinner("Retrieving relevant chunks and asking the LLM..."):
            try:
                result = answer_with_citations(query, ns)
                answer = result.get("answer", "I could not find it")
                citations = result.get("citations", [])
                meta = {"time": datetime.utcnow().isoformat(), "source_count": len(citations)}
                st.session_state.messages.append(("assistant", answer, meta))
                # display immediate answer
                st.experimental_rerun()
            except Exception as e:
                st.session_state.messages.append(("assistant", f"Error: {e}", {"time": datetime.utcnow().isoformat()}))
                st.experimental_rerun()

# right column: sources, controls, export
with right_col:
    st.markdown("## 🔎 Sources & Controls")
    selected_ns = st.session_state.get("namespace", namespace_input)
    st.markdown(f"**Active namespace:** `{selected_ns}`")
    st.markdown("### Last answer sources")
    # show sources for the last assistant message if available
    last_assistant = None
    for r, t, m in reversed(st.session_state.messages):
        if r == "assistant":
            last_assistant = (t, m)
            break

    if last_assistant:
        # to show the actual snippets, we ask answer_with_citations again but it returns citations
        try:
            # call again to fetch citations (cheap compared to whole pipeline)
            _, citations = (last_assistant[0], [])  # placeholder
            # Instead of double-calling the chain, store citations in messages in future
            # For now, try to re-run answer_with_citations to fetch citations (safe)
            q_text = None
            # find the user query immediately before the assistant message
            for i in range(len(st.session_state.messages)-1):
                if st.session_state.messages[i][0] == "user":
                    possible_q = st.session_state.messages[i][1]
                    # heuristic: last user message before assistant
                    if i >= len(st.session_state.messages)-2:
                        q_text = possible_q
            if q_text:
                res = answer_with_citations(q_text, selected_ns)
                cites = res.get("citations", [])
            else:
                cites = []
        except Exception:
            cites = []

        if cites:
            for i, c in enumerate(cites, start=1):
                st.markdown(f"**[{i}] p.{c['page']} — {c['source']}**")
                st.caption(c.get("snippet", "")[:400])
        else:
            st.markdown("_No citations available. If the assistant returned 'I could not find it', the answer is not in the book._")
    else:
        st.markdown("_No assistant answers yet._")

    st.markdown("---")
    st.markdown("### Export / Save Chat")
    if st.button("Export chat as JSON"):
        export = {
            "exported_at": datetime.utcnow().isoformat(),
            "namespace": st.session_state.get("namespace"),
            "messages": [
                {"role": r, "text": t, "meta": m}
                for (r, t, m) in st.session_state.messages
            ],
        }
        st.download_button("Download chat JSON", data=json.dumps(export, indent=2), file_name="chat_export.json", mime="application/json")

    st.markdown("---")
    st.markdown("### Help / Tips")
    st.markdown("- Index a PDF first (sidebar).")
    st.markdown("- If the bot cannot find an answer, it will return: **I could not find it**.")
    st.markdown("- Keep your `.env` private; do not commit keys to GitHub.")

