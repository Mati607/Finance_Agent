from Ingestion import DataObject, Ingest, MimeType
from Chunking import Chunk, Chunker
from Retrieval import SemanticContext, NumericContext, Retriever
from pprint import pprint
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    file_path = "commentary_excerpts.txt"
    mime = MimeType.TXT
    query = "Emerging Markets delivered $9.1M in Q3, essentially in line with the $9.0M plan. This marks the first quarter of consistent plan attainment since the LATAM underperformance in Q1. EBITDA of $1.4M was $0.1M favorable."

    ans = Retriever([file_path],[mime],1,Ingest, Chunker).get_semantic_context(query)
    for item in ans:
        chunk = {k: v for k, v in item.chunk.__dict__.items() if k not in ("_embedding", "client")}
        data = {**item.__dict__, "chunk": chunk}
        pprint(data)
