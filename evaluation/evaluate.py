"""
Run the Finance Agent against evaluation/eval_set.json and grade it.

- Graded questions are scored with RAGAS (faithfulness, answer_relevancy,
  context_precision, context_recall), using Gemini as both judge LLM and
  embedding model so no OpenAI credits are needed.
- "expect_refusal" questions bypass RAGAS and are spot-checked for a
  "Cannot be answered" response.

Usage:
    python -m evaluation.evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from Agent import Agent
from Chunking import Chunker
from Ingestion import Ingest, MimeType
from Retrieval import Retriever

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = Path(__file__).resolve().parent / "eval_set.json"

CORPUS = [
    (REPO_ROOT / "commentary_excerpts.txt", MimeType.TXT),
    (REPO_ROOT / "bu_financials_q3.csv", MimeType.CSV),
]


def build_agent() -> Agent:
    for path, _ in CORPUS:
        if not path.exists():
            sys.exit(f"Missing corpus file: {path}. See README step 3.")
    paths = [str(p) for p, _ in CORPUS]
    mimes = [m for _, m in CORPUS]
    retriever = Retriever(
        paths,
        mimes,
        topk=3,
        ingestor=Ingest,
        chunker=Chunker,
        topk_retrieve=10,
        topk_final=3,
    )
    return Agent(retriever)


def contexts_from_response(resp) -> list[str]:
    """Flatten semantic + numeric citations into strings for RAGAS."""
    out: list[str] = []
    for c in resp.semantic_citations:
        out.append(c.chunk.content)
    for n in resp.numeric_citations:
        if not n.table.empty:
            out.append(f"SQL: {n.sql}\nrows:\n{n.table.to_csv(index=False)}")
    return out


def run_ragas(samples: list[dict]) -> None:
    """Grade samples with RAGAS using Gemini. Imports live inside the
    function so the refusal-only path doesn't pull in heavy deps."""
    from datasets import Dataset
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(samples)

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print("\n=== RAGAS scores ===")
    print(result)
    print("\n=== Per-question breakdown ===")
    print(result.to_pandas().to_string(index=False))


def main() -> None:
    eval_items = json.loads(EVAL_FILE.read_text())
    agent = build_agent()

    rag_samples: list[dict] = []
    refusal_results: list[dict] = []

    for item in eval_items:
        qid = item["id"]
        q = item["question"]
        print(f"\n[{qid}] {q}")
        resp = agent.generate_response(q)
        answer = resp.response
        contexts = contexts_from_response(resp)
        print(f"  answer: {answer[:240]}{'…' if len(answer) > 240 else ''}")

        if item.get("expect_refusal"):
            refused = answer.strip().lower().startswith("cannot be answered")
            refusal_results.append({"id": qid, "refused": refused, "answer": answer})
        else:
            rag_samples.append(
                {
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item["ground_truth"],
                }
            )

    print("\n=== Refusal checks ===")
    if not refusal_results:
        print("(none)")
    for r in refusal_results:
        status = "PASS" if r["refused"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if not r["refused"]:
            print(f"  got: {r['answer']}")

    if rag_samples:
        run_ragas(rag_samples)


if __name__ == "__main__":
    main()
