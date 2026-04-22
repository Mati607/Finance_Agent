from Ingestion import Ingest, MimeType
from Chunking import Chunker, Chunk
import json
from typing import List, Optional, Tuple
from google import genai
import numpy as np
import pandas as pd
import faiss
import duckdb
import os

from baml_client.sync_client import b
from baml_client.types import TableInfo, RerankCandidate

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
            sql: str,
            filename: str
    ):
        self.table = selected_rows
        self.sql = sql
        self.source_file = filename


class Retriever:
    def __init__(
            self,
            file_urls: List[str],
            mime_types: MimeType,
            ingestor: Ingest,
            chunker: Chunker,
            topk_retrieve: int,
            topk_final: int,
    ):
        self.file_urls = file_urls
        self.mime_types = mime_types
        self.topk_retrieve = topk_retrieve
        self.topk_final = topk_final
        self.ingestor = ingestor
        self.chunker = chunker
        self.text_tables = []
        self.data_tables = []
        self.chunks = []
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

        k = min(self.topk_retrieve, len(self.chunks))
        if k <= 0:
            return []

        scores, indices = self.index.search(query_emb, k)

        candidates: List[SemanticContext] = []
        for j, i in enumerate(indices[0]):
            if i < 0:
                continue
            candidates.append(SemanticContext(self.chunks[i], float(scores[0][j])))

        return self._rerank(query, candidates)

    def _rerank(
            self,
            query: str,
            candidates: List["SemanticContext"],
    ) -> List["SemanticContext"]:
        if not candidates:
            return []

        payload: List[RerankCandidate] = []
        for idx, ctx in enumerate(candidates):
            chunk = ctx.chunk
            payload.append(
                RerankCandidate(
                    index=idx,
                    source_file=str(chunk.filename or ""),
                    title=str(chunk.title or ""),
                    author=str(chunk.author or ""),
                    participants=", ".join(chunk.participants or []),
                    date=str(chunk.date or ""),
                    content=chunk.content,
                )
            )

        try:
            ranked = b.RerankChunks(user_query=query, candidates=payload)
        except Exception as e:
            print(f"[retriever] rerank failed, falling back to dense order: {e}")
            return candidates[: self.topk_final]

        scored: List[Tuple[int, float]] = []
        seen = set()
        for item in ranked:
            idx = item.index
            if idx < 0 or idx >= len(candidates) or idx in seen:
                continue
            seen.add(idx)
            scored.append((idx, float(item.score)))

        scored.sort(key=lambda p: p[1], reverse=True)

        top = []
        for idx, score in scored[: self.topk_final]:
            ctx = candidates[idx]
            ctx.score = score
            top.append(ctx)
        return top

    def _generate_sql_query(self, query: str) -> List[Tuple[str, int]]:
        if not self.data_tables:
            return []

        n = len(self.data_tables)
        table_infos: List[TableInfo] = []
        for i, table in enumerate(self.data_tables):
            prep = table.data.reset_index().rename(columns={"index": "row_id"})
            columns = {"columns": [{"name": c, "dtype": str(prep[c].dtype)} for c in prep.columns]}
            sample_csv = prep.head(5).to_csv(index=False)
            stored_schema = table.schema
            if isinstance(stored_schema, tuple) and len(stored_schema) > 1:
                stored_schema = stored_schema[1]
            if not isinstance(stored_schema, dict):
                stored_schema = {}
            table_infos.append(
                TableInfo(
                    table_index=i,
                    filename=str(table.filename),
                    ingested_schema=json.dumps(stored_schema),
                    columns=json.dumps(columns),
                    sample_rows=sample_csv,
                )
            )

        results = b.GenerateSQL(user_query=query, tables=table_infos)

        out: List[Tuple[str, int]] = []
        seen = set()
        for item in results:
            if not item.relevant:
                continue
            sql = (item.sql or "").strip()
            if not sql:
                continue
            idx = item.table_index
            if idx < 0 or idx >= n or idx in seen:
                continue
            seen.add(idx)
            out.append((sql, idx))

        out.sort(key=lambda p: p[1])
        return out

    def get_numeric_context(self, query: str) -> Optional[List[NumericContext]]:
        pairs = self._generate_sql_query(query)
        if not pairs:
            return None

        results = []
        for sql, table_index in pairs:
            table = self.data_tables[table_index]
            df = table.data.reset_index().rename(columns={"index": "row_id"})
            con = duckdb.connect()
            con.register("df", df)
            try:
                selected = con.execute(sql).df()
            except duckdb.Error as e:
                print(f"[retriever] skipping table {table_index}: bad SQL: {e}")
                continue
            if "row_id" in selected.columns:
                selected = selected.set_index("row_id")
            results.append(NumericContext(selected, sql, table.filename))

        return results
