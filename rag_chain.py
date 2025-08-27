# rag_chain.py
import os
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LLM providers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-book-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

STRICT_REFUSAL = "I could not find it"

SYSTEM_PROMPT = (
    "You are a librarian bot. Answer ONLY using the provided book context. "
    f"If the answer is not fully present in the context, reply exactly: {STRICT_REFUSAL}. "
    "Keep answers concise, cite pages."
)

# ---------- Embeddings ----------

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ---------- Pinecone Setup ----------

pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_pinecone_index():
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX)
# ---------- Utils ----------

def slug_namespace(title: str) -> str:
    s = title.strip().lower().replace(" ", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "-_")
    if not s:
        s = hashlib.sha1(title.encode()).hexdigest()[:8]
    return s

def format_docs_for_prompt(docs: List[Any]) -> str:
    blocks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        src = os.path.basename(d.metadata.get("source", "book.pdf"))
        blocks.append(f"[p.{page} | {src}]\n{d.page_content}")
    return "\n\n".join(blocks)

# ---------- Ingestion ----------

def ingest_pdf(pdf_path: str, namespace: str) -> Dict[str, Any]:
    """Load a PDF, chunk it, and upsert to Pinecone under the given namespace."""
    ensure_pinecone_index()

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # each page is a Document with metadata {source, page}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    embeddings = get_embeddings()
    vector_store = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=PINECONE_INDEX,
        namespace=namespace,
        text_key="text",
    )

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "namespace": namespace,
        "index": PINECONE_INDEX,
    }

# ---------- LLM Factory ----------

def get_llm():
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing in .env")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY missing in .env")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    elif LLM_PROVIDER == "ollama":
        # Lightweight placeholder via OpenAI-compatible path is possible,
        # but for simplicity instruct user to expose via langchain_community.llms.Ollama if needed.
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

# ---------- Retrieval + RAG Chain ----------

def ensure_or_refuse(docs: List[Any]) -> str:
    if not docs:
        return ""
    return format_docs_for_prompt(docs)

# ⚠️ build_rag_chain function seems missing in your code
# I assume you meant something like this:

def build_rag_chain(namespace: str):
    llm = get_llm()
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Retrieval setup
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
        namespace=namespace,
        text_key="text",
    )

    chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(lambda q: ensure_or_refuse(vector_store.similarity_search(q, k=6))),
        }
        | prompt
        | llm
        | parser
    )

    return chain, vector_store

def answer_with_citations(query: str, namespace: str) -> Dict[str, Any]:
    chain, vector_store = build_rag_chain(namespace)
    docs = vector_store.similarity_search(query, k=6)

    if not docs:
        return {"answer": STRICT_REFUSAL, "citations": []}

    answer = chain.invoke(query)

    # Guardrail: enforce refusal if no citations
    normalized = answer.strip().lower()
    if STRICT_REFUSAL.lower() not in normalized and all(
        key_phrase.lower() not in normalized for key_phrase in ["page", "p."]
    ) and not docs:
        return {"answer": STRICT_REFUSAL, "citations": []}

    citations = []
    for d in docs[:4]:  # top 4 citations
        citations.append(
            {
                "page": d.metadata.get("page", "?"),
                "source": os.path.basename(d.metadata.get("source", "GenAI-AWS.pdf")),
                "snippet": (d.page_content[:280] + "…") if len(d.page_content) > 280 else d.page_content,
            }
        )

    if answer.strip() == "":
        answer = STRICT_REFUSAL

    return {"answer": answer, "citations": citations}
