class SemanticEvidence {
  source_file string
  title string
  author string
  participants string
  date string
  content string
}

class NumericEvidence {
  source_file string
  sql string
  rows_csv string
}

class FinancialAnswer {
  answerable bool
  answer string
}

function AnswerFinancialQuery(
  user_query: string,
  semantic_evidence: SemanticEvidence[],
  numeric_evidence: NumericEvidence[]
) -> FinancialAnswer {
  client GeminiDefault
  prompt #"
    You are a meticulous financial analysis assistant. Answer the user's
    question using ONLY the evidence provided below. This is a financial
    setting: precision and faithful attribution matter. Do not speculate,
    do not fabricate numbers or facts, and do not rely on outside knowledge
    or memory.

    Rules:
    - If the provided evidence does not contain enough information to answer
      the question confidently, set `answerable` to false and set `answer` to
      a short message beginning with "Cannot be answered" that explains what
      specifically is missing. Do not guess or partially answer.
    - Entity attribution is critical. Every statement you quote or paraphrase
      must be attributed to the exact author / participant / source shown in
      the evidence. Never mix or blend statements across speakers. A person's
      own statement is not independent evidence about that person — e.g. if
      the user asks what the CFO did, said, or decided, only the CFO's own
      recorded statements (or documents authored by the CFO) qualify, and
      another speaker's words cannot be substituted for the CFO's. If the
      requested entity is not present in the evidence, answer "Cannot be
      answered".
    - Do not invent numbers, dates, business units, product names, or people.
      Every concrete fact in the answer must be grounded in the evidence.
    - Prefer numeric evidence (actual table rows) for quantitative questions.
      Prefer semantic evidence (commentary / transcripts) for qualitative or
      "why" questions. Combine both when the question requires it.
    - If numeric evidence and semantic commentary disagree, surface the
      discrepancy rather than picking one silently.
    - Keep the answer concise and direct. Cite the source file(s) and, for
      semantic evidence, the author or participants you relied on.

    USER QUESTION:
    {{ user_query }}

    SEMANTIC EVIDENCE (commentary / transcripts):
    {% if semantic_evidence|length == 0 %}
    (none)
    {% else %}
    {% for s in semantic_evidence %}
    --- source_file={{ s.source_file }} | title={{ s.title }} | author={{ s.author }} | participants={{ s.participants }} | date={{ s.date }} ---
    {{ s.content }}
    {% endfor %}
    {% endif %}

    NUMERIC EVIDENCE (tabular query results):
    {% if numeric_evidence|length == 0 %}
    (none)
    {% else %}
    {% for n in numeric_evidence %}
    --- source_file={{ n.source_file }} ---
    sql: {{ n.sql }}
    rows (CSV):
    {{ n.rows_csv }}
    {% endfor %}
    {% endif %}

    {{ ctx.output_format }}
  "#
}
