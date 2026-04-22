class TableInfo {
  table_index int
  filename string
  ingested_schema string
  columns string
  sample_rows string
}

class TableSql {
  table_index int
  relevant bool
  sql string
}

function GenerateSQL(user_query: string, tables: TableInfo[]) -> TableSql[] {
  client GeminiDefault
  prompt #"
    You help answer a user question using tabular data.
    Each table block below represents one distinct table (they may have different schemas).

    For each table_index, decide if that table alone can answer the user question
    using a DuckDB SELECT (or WITH ... SELECT) over a relation named exactly `df`.
    Assume `df` has all the columns listed plus an integer `row_id` column.

    Rules:
    - If a table is irrelevant or cannot answer the question from its columns and
      sample rows, set `relevant` to false and `sql` to an empty string.
    - If a table can answer the question, set `relevant` to true and provide a
      single valid DuckDB SQL query that references only the relation `df`.
    - SELECT every column needed for a downstream LLM to justify the answer,
      not just the column(s) used for filtering or ordering. For the
      dimension column(s) the question is about (e.g. business unit, quarter),
      always include them AND every supporting metric column that makes the
      answer verifiable — actual, plan, variance, percentage, headcount, notes,
      etc. When in doubt, include more columns rather than fewer. Do not use
      `SELECT *` unless the table is narrow; otherwise pick the relevant
      columns explicitly so the evidence is self-contained.
    - Preserve the original column names exactly as shown (including spaces,
      units like "($M)", and casing) — quote them with double quotes when
      needed.
    - Return exactly one entry per table, preserving the original table_index.
    - Do not use markdown fences or commentary in the `sql` field.

    USER QUESTION:
    {{ user_query }}

    TABLES:
    {% for t in tables %}
    --- table_index={{ t.table_index }} (file={{ t.filename }}) ---
    When this table is used, DuckDB exposes it as the relation named exactly `df`.
    ingested_schema: {{ t.ingested_schema }}
    columns: {{ t.columns }}
    sample_rows (first 5, CSV):
    {{ t.sample_rows }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
