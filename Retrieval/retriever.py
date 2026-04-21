from Ingestion import Ingest, MimeType
from Chunking import Chunker, Chunk
from typing import List
from google import genai
import numpy as np
import pandas as pd
import faiss
import duckdb
import os

class SemanticContext:
    def __init__(
            self,
            chunk: Chunk,
            score: float
    ):
        self.chunk = chunk
        self.score = score

class NumericContext:
    def __init__(
            self,
            selected_rows: pd.DataFrame,
            filename: str
    ):
        self.table = selected_rows
        self.source_file = filename


class Retriever:
    def __init__(
            self,
            file_urls: List[str],
            mime_types: MimeType,
            topk: int,
            ingestor: Ingest,
            chunker: Chunker
    ):
        self.file_urls = file_urls
        self.mime_types = mime_types
        self.topk = topk
        self.ingestor = ingestor
        self.chunker = chunker
        self.text_tables = []
        self.data_tables = []
        self.chunks = []
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        self._get_tables()
        self._get_chunks()
        self.index = self._build_faiss_index()

    
    def _get_tables(self):
        for file, typ in zip(self.file_urls,self.mime_types):
            table = self.ingestor(file,typ).execute()
            if table.document_type == MimeType.TXT:
                self.text_tables.append(table)
            if (
                table.document_type == MimeType.CSV or
                table.document_type == MimeType.XLSX
            ):
                self.data_tables.append(table)
            
    
    def _get_chunks(self):
        for table in self.text_tables:
            chunks = self.chunker(table).txt_splitter()
            self.chunks.extend(chunks)


    def _embed_user_query(self,query:str):
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        return np.array(response.embeddings[0].values, dtype="float32")
    
    def _build_faiss_index(self):
        embeddings = [x.embedding for x in self.chunks]
        embeddings = np.array(embeddings).astype("float32")

        norms = np.linalg.norm(embeddings,axis=1,keepdims=True)
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index
    
    def get_semantic_context(self,query):
        query_emb = self._embed_user_query(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.reshape(1,-1)

        scores, indices = self.index.search(query_emb,self.topk)

        results = []
        for j,i in enumerate(indices[0]):
            res = SemanticContext(self.chunks[i],scores[0][j])
            results.append(res)

        return results
   
    def get_numeric_context(self, sql: str, table_index: int):
        df = self.data_tables[table_index]
        df = df.reset_index().rename(columns={"index": "row_id"})

        con = duckdb.connect()
        con.register("df", df)

        result = con.execute(sql).df()

        if "row_id" in result.columns:
            result = result.set_index("row_id")

        return result