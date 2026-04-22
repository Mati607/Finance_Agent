# Finance Agent — RAG over mixed financial data

A retrieval-augmented QA system over a mixed corpus of structured quarterly
financials (CSV/XLSX) and unstructured commentary (earnings-call-style
transcripts, BU review memos). Answers variance questions with grounded
citations, and refuses rather than fabricates when the corpus does not
support an answer.

## How to run

**Prerequisites:** Python 3.10+, a Gemini API key.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the API key

Create a `.env` file in the repo root:

```
GOOGLE_API_KEY=<your-gemini-key>
```

### 4. Start the server

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in a browser.

### 5. Use the app

1. Upload `bu_financials_q3.csv` and `commentary_excerpts.txt` via the UI.
2. Ask a question — e.g. *"Which business unit had the largest unfavorable variance in Q3, and what factors does the commentary attribute this to?"*
3. The response shows the answer plus the semantic and numeric citations it relied on.

## Repository layout

| Path | Purpose |
|------|---------|
| [Ingestion/process.py](Ingestion/process.py) | Typed file ingestion → `DataObject` (CSV/XLSX → `DataFrame`, TXT → single-row frame). |
| [Chunking/chunking.py](Chunking/chunking.py) | Metadata-aware splitter for the commentary corpus; lazy-embeds chunks via Gemini. |
| [Retrieval/retriever.py](Retrieval/retriever.py) | FAISS semantic index + LLM-generated DuckDB SQL over structured tables. |
| [Agent/agent.py](Agent/agent.py) | Orchestrates retrieval, builds evidence, calls BAML, returns citations. |
| [baml_src/](baml_src/) | BAML prompts — `AnswerFinancialQuery` and `GenerateSQL`. |
| [app.py](app.py) | FastAPI server with upload / reset / query endpoints. |

## Pipeline design

### Ingestion

Each source becomes a `DataObject` carrying a pandas `DataFrame`, a
column schema, filename, and `MimeType`. Text files are stored as a
single-row frame so the same object model flows through the rest of the
pipeline uniformly.

### Chunking strategy (unstructured)

See [Chunking/chunking.py](Chunking/chunking.py). The commentary corpus
uses `--- DOCUMENT N ---` markers followed by a
`Source: / Date: / Author: / Participants:` header block. The chunker:

1. **Splits on document markers** so each logical document keeps its own
   metadata namespace.
2. **Parses the metadata header** into structured fields (`title`,
   `author`, `participants`, `date`) stamped on every chunk from that
   document — this is what makes entity attribution work later.
3. **Splits the body into paragraphs** (blank-line delimited). Paragraph
   boundaries are already semantically meaningful here (transcript turns,
   memo points), so this preserves coherence without stapling unrelated
   ideas together.
4. **Preserves exact character offsets** so a citation can be resolved
   back to the exact span in the source file.

### Embedding approach

Google `gemini-embedding-001` for both chunks and user queries. Each
chunk is serialized with a header prefix
(`[#Chunk index: N | title | date | by author]\ncontent`) before
embedding, so metadata participates in similarity. This boosts recall on
entity-sensitive queries (e.g. *"what did the BU Finance Lead say?"*).

### Retrieval method

Two parallel retrievers, one per modality:

- **Semantic (unstructured):** FAISS `IndexFlatIP` with L2-normalised
  vectors (inner product == cosine similarity). Top-k=5 by default.
  [Retrieval/retriever.py:85-109](Retrieval/retriever.py#L85-L109).
- **Numeric (structured):** the user query + every table's schema and a
  5-row sample are passed to `GenerateSQL` (BAML). The LLM decides
  per-table whether it is relevant and emits a DuckDB `SELECT` over a
  relation aliased `df`. Only relevant tables returning non-empty results
  become numeric citations.
  [Retrieval/retriever.py:111-171](Retrieval/retriever.py#L111-L171).

Both paths run on every query. If **both** return nothing, the agent
short-circuits with a deterministic `Cannot be answered` before ever
calling the answering LLM
([Agent/agent.py:76-84](Agent/agent.py#L76-L84)).

### Grounding / citation

Every response carries structured citations:

- `semantic_citations`: source file, title/author/participants/date, the
  literal content string, similarity score, and character offsets.
- `numeric_citations`: source CSV/XLSX filename, the exact DuckDB SQL
  that was executed, and the rows returned.

A reviewer can re-run the SQL and/or open the source file at the cited
offsets — no part of the answer is unattributable.

## Example output

**Query:** *"Which business unit had the largest unfavorable variance in Q3, and what factors does the commentary attribute this to?"*

**Answer (abridged):**
> Enterprise Solutions had the largest unfavorable Q3 variance: revenue
> of $39.2M vs. $46.0M plan (−$6.8M, −14.8%) and EBITDA of $5.1M vs.
> $9.2M plan (−$4.1M). Per the Q3 2024 Enterprise Solutions Business
> Review (authored by the BU Finance Lead), the miss is attributed to
> (1) a ~$4.2M FSI renewal that slipped from late September into Q4 due
> to a customer procurement freeze tied to a pending acquisition, and
> (2) softer professional services revenue as prior-quarter
> implementations wound down. Headcount was 19 above plan because an
> August hiring cohort was still ramping.

**Citations returned:**
- *Numeric*: `bu_financials_q3.csv` — `SELECT ... WHERE Quarter = 'Q3 2024' ORDER BY "Revenue_Variance ($M)" ASC` (the Enterprise row is the most negative).
- *Semantic*: `commentary_excerpts.txt`, Document 1, *Internal Business Review — Q3 2024, Enterprise Solutions BU*, author *BU Finance Lead*, date *October 14, 2024*.

## Fallback behaviour (no fabrication)

Three layers, in order:

1. **Retrieval layer** — if both FAISS and the SQL planner return
   nothing, the agent returns a deterministic `Cannot be answered` string
   without calling the answering LLM.
2. **Prompt layer** — the `AnswerFinancialQuery` prompt in
   [baml_src/answer_generation.baml](baml_src/answer_generation.baml)
   instructs the model to set `answerable=false` and begin its answer
   with `"Cannot be answered"` if evidence is insufficient, and forbids
   mixing speakers or inventing numbers, dates, entities, or people.
3. **Schema layer** — `FinancialAnswer` has an explicit `answerable: bool`,
   so downstream consumers can gate on it programmatically rather than
   string-sniffing.

---

## Part 2 — Hallucination mitigation

**Probe query:** *"What was the CEO's guidance on the Q3 variance and what remediation steps did she commit to?"*

### Failure modes

This query is a honeypot: the corpus contains **no CEO**. The speakers
in the Q3 QBR transcript (Document 2) are the CFO, VP Sales, BU Finance
Lead, and FP&A. Documents 1, 3, 4 are authored by BU finance staff;
Document 5 is an unsigned FP&A summary. The question also presupposes
**remediation commitments** that nobody in the corpus actually makes.

1. **Speaker confusion / role substitution.** A naive RAG will retrieve
   Document 2 (it's the most query-similar chunk — it talks about Q3
   variance and guidance) and then blur "the CFO said X" into "the CEO
   said X". The gendered *she* in the query nudges the model toward
   picking *any* senior speaker and mapping them onto that pronoun.
2. **Commitment fabrication.** The CFO asks FP&A to reflect the FSI deal
   and flag services risk in the Q4 bridge — an *internal ask*, not a
   public remediation commitment. A loose prompt will paraphrase this as
   "she committed to corrective steps including X, Y, Z," inventing
   structure that isn't there. Document 5 also explicitly disclaims that
   it is not complete — which models routinely ignore.

### Mitigation implemented

Prompt-level entity-attribution constraints in
[baml_src/answer_generation.baml](baml_src/answer_generation.baml)
combined with a structured `answerable: bool` output and the
retrieval-layer empty-evidence short-circuit in
[Agent/agent.py:76-84](Agent/agent.py#L76-L84). The prompt explicitly
says another speaker's words cannot be substituted for the requested
entity's, and that if the entity is not present in the evidence the
answer must be `Cannot be answered`. The `answerable` boolean means
consumers don't have to string-match the refusal text.

### Tradeoff

This catches the two failure modes above cleanly: the CEO is not a
participant or author anywhere in the evidence, so the model correctly
refuses rather than laundering the CFO's statements into CEO quotes. It
also catches the commitment-fabrication case because the prompt forbids
inventing concrete facts, and there are no CEO-authored commitments to
quote.

What it **misses**: prompt-only mitigations are probabilistic — a
sufficiently confused model can still regress under adversarial phrasing
(*"summarise leadership's view on Q3"*), because *leadership* is fuzzy
enough to license quoting the CFO. It also does not catch the inverse
failure, where a real entity *is* present but the retriever fails to
surface their chunk in top-k, producing a false-negative refusal. A
deterministic fix — an entity-presence pre-check against chunk metadata
(`author` / `participants`) before the LLM is called, plus a
cross-encoder reranker to protect recall — would close both gaps but
adds latency and a second failure surface (role normalisation: is
"Chief Executive" the CEO?). For this corpus size the prompt-level
mitigation is the right cost/benefit point; at scale I would add the
metadata pre-check.
