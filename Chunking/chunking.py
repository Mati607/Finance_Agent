import re
import uuid
from Ingestion import DataObject
from google import genai
import os



class Chunk:
    def __init__(
            self,
            id: uuid.UUID,
            content: str,

            filename: str,
            document_no: int,
            title: str,
            author: str,
            participants: list[str],
            date: str,

            chunk_index: int,
            start_char: int,
            end_char: int,
    ):
        self.id = id
        self.content = content
        self.filename = filename
        self.document_no = document_no
        self.title = title
        self.author = author
        self.participants = participants
        self.date = date
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self._embedding = None

    def to_string(self) -> str:
        header_bits = [f"#Chunk index: {self.chunk_index}"]
        if self.title:
            header_bits.append(self.title)
        if self.date:
            header_bits.append(self.date)
        if self.author:
            header_bits.append(f"by {self.author}")
        elif self.participants:
            header_bits.append(f"participants: {', '.join(self.participants)}")

        header = " | ".join(header_bits)
        return f"[{header}]\n{self.content}"
    
    @property
    def embedding(self):
        if self._embedding is not None:
            return self._embedding
        text = self.to_string()
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        self._embedding = response.embeddings[0].values
        return self._embedding



class Chunker:
    _DOC_SPLIT_RE = re.compile(r"^---\s*DOCUMENT\s+(\d+)\s*---\s*$", re.MULTILINE)
    _DASH_LINE_RE = re.compile(r"^---\s*$", re.MULTILINE)
    _META_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z ]*?):\s*(.+?)\s*$")
    _PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]*)*")

    def __init__(self, data: DataObject):
        self.data = data

    def _parse_header(self, header_text: str) -> dict:
        meta = {}
        for line in header_text.splitlines():
            m = self._META_LINE_RE.match(line.strip())
            if m:
                meta[m.group(1).strip().lower()] = m.group(2).strip()
        return meta

    def _split_sub_documents(self, text: str):
        """
        Yield (document_no, metadata_dict, body_text, body_offset) for each section.
        body_offset is the absolute char offset of body_text[0] in the original text.
        """
        matches = list(self._DOC_SPLIT_RE.finditer(text))
        if not matches:
            yield 1, {}, text, 0
            return

        for i, m in enumerate(matches):
            doc_no = int(m.group(1))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end]

            header_end = self._DASH_LINE_RE.search(section)
            if header_end:
                header_text = section[:header_end.start()]
                body = section[header_end.end():]
                body_offset = start + header_end.end()
            else:
                header_text, body, body_offset = "", section, start

            yield doc_no, self._parse_header(header_text), body, body_offset

    def _iter_paragraphs(self, body: str, body_offset: int):
        """Yield (paragraph_text, abs_start, abs_end) for each non-empty paragraph."""
        for m in self._PARAGRAPH_RE.finditer(body):
            raw = m.group(0)
            para = raw.strip()
            if not para or re.fullmatch(r"-+", para):
                continue
            lead = len(raw) - len(raw.lstrip())
            abs_start = body_offset + m.start() + lead
            abs_end = abs_start + len(para)
            yield para, abs_start, abs_end

    def txt_splitter(self) -> list[Chunk]:
        text = self.data.data['content'][0]
        filename = self.data.filename

        chunks: list[Chunk] = []

        for doc_no, meta, body, body_offset in self._split_sub_documents(text):
            participants = [
                p.strip() for p in meta.get("participants", "").split(",") if p.strip()
            ]

            for idx, (content, s, e) in enumerate(self._iter_paragraphs(body, body_offset)):
                chunks.append(
                    Chunk(
                        id=uuid.uuid4(),
                        content=content,
                        filename=filename,
                        document_no=doc_no,
                        title=meta.get("source", ""),
                        author=meta.get("author", ""),
                        participants=participants,
                        date=meta.get("date", ""),
                        chunk_index=idx,
                        start_char=s,
                        end_char=e,
                    )
                )

        return chunks