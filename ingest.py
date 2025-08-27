# ingest.py
import argparse
import os
from rag_chain import ingest_pdf, slug_namespace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF book into Pinecone")
    parser.add_argument("pdf", help="Path to the PDF (200+ pages recommended)")
    parser.add_argument("--name", help="Custom namespace name (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(args.pdf)

    title = args.name or os.path.splitext(os.path.basename(args.pdf))[0]
    ns = slug_namespace(title)

    stats = ingest_pdf(args.pdf, ns)
    print(
        f"Ingested {stats['chunks']} chunks from {stats['pages']} pages "
        f"into namespace '{ns}' on index '{stats['index']}'."
    )
