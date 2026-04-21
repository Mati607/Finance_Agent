from typing import Tuple
from Retrieval import Retriever, SemanticContext, NumericContext
from baml_client.sync_client import b
from baml_client.types import (
    SemanticEvidence,
    NumericEvidence,
)


class Response:
    def __init__(
            self,
            user_response: str,
            semantic_citations: list[SemanticContext],
            numeric_citations: list[NumericContext],
    ):
        self.response = user_response
        self.semantic_citations = semantic_citations
        self.numeric_citations = numeric_citations


class Agent:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def _to_semantic_evidence(
            self, semantic_context: list[SemanticContext],
    ) -> list[SemanticEvidence]:
        evidence = []
        for item in semantic_context:
            chunk = item.chunk
            evidence.append(
                SemanticEvidence(
                    source_file=str(chunk.filename or ""),
                    title=str(chunk.title or ""),
                    author=str(chunk.author or ""),
                    participants=", ".join(chunk.participants or []),
                    date=str(chunk.date or ""),
                    content=chunk.content,
                )
            )
        return evidence

    def _to_numeric_evidence(
            self, numeric_context: list[NumericContext],
    ) -> list[NumericEvidence]:
        evidence = []
        for item in numeric_context:
            evidence.append(
                NumericEvidence(
                    source_file=str(item.source_file or ""),
                    sql=item.sql,
                    rows_csv=item.table.to_csv(),
                )
            )
        return evidence

    def compose_context(
            self,
            semantic_context: list[SemanticContext],
            numeric_context: list[NumericContext],
    ) -> Tuple[list[SemanticEvidence], list[NumericEvidence]]:
        return (
            self._to_semantic_evidence(semantic_context or []),
            self._to_numeric_evidence(numeric_context or []),
        )

    def generate_response(self, query: str) -> Response:
        semantic_context = self.retriever.get_semantic_context(query) or []
        numeric_context = self.retriever.get_numeric_context(query) or []

        semantic_evidence, numeric_evidence = self.compose_context(
            semantic_context, numeric_context,
        )

        if not semantic_evidence and not numeric_evidence:
            return Response(
                user_response=(
                    "Cannot be answered: no relevant semantic or numeric "
                    "evidence was retrieved from the available sources."
                ),
                semantic_citations=semantic_context,
                numeric_citations=numeric_context,
            )

        result = b.AnswerFinancialQuery(
            user_query=query,
            semantic_evidence=semantic_evidence,
            numeric_evidence=numeric_evidence,
        )

        return Response(
            user_response=result.answer,
            semantic_citations=semantic_context,
            numeric_citations=numeric_context,
        )
