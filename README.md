# ==== Core ====
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_INDEX=rag-book-index
PINECONE_CLOUD=aws # aws or gcp (serverless)
PINECONE_REGION=us-east-1 # pick your region (e.g., us-east-1)


# Embeddings
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384


# LLM provider: groq | openai | ollama
LLM_PROVIDER=groq


# If LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_KEY


# If LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_OPENAI_KEY


# If LLM_PROVIDER=ollama (local)
OLLAMA_MODEL=llama3.1