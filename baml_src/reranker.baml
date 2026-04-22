class RerankCandidate {
  index int
  source_file string
  title string
  author string
  participants string
  date string
  content string
}

class RerankedChunk {
  index int
  score float
  reason string
}

function RerankChunks(
  user_query: string,
  candidates: RerankCandidate[]
) -> RerankedChunk[] {
  client GeminiDefault
  prompt #"
    You are a precision-focused reranker for a financial QA system.
    Given the user's question and a list of candidate text chunks retrieved
    from source documents, score how directly each chunk answers the question.

    Scoring rubric (float in [0.0, 1.0]):
    - 1.0  The chunk contains the exact fact(s) or statement(s) needed to
           answer the question, attributed to the right entity.
    - 0.7  The chunk is clearly on-topic and contributes supporting evidence
           but does not fully answer on its own.
    - 0.4  The chunk is tangentially related (same business unit, same
           period, adjacent topic) but does not directly address the question.
    - 0.0  The chunk is off-topic, or only superficially matches keywords.

    Be strict. In a financial setting, retrieving an almost-right chunk is
    worse than retrieving fewer, truly-relevant chunks. Entity attribution
    matters: if the question asks what person X said or did, chunks by or
    about someone else should be scored low even if the topic matches.

    Return exactly one entry per candidate, preserving the original `index`.
    Keep `reason` to one short sentence.

    USER QUESTION:
    {{ user_query }}

    CANDIDATES:
    {% for c in candidates %}
    --- index={{ c.index }} | source_file={{ c.source_file }} | title={{ c.title }} | author={{ c.author }} | participants={{ c.participants }} | date={{ c.date }} ---
    {{ c.content }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
