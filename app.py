import os
import pathlib
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from Ingestion import Ingest, MimeType
from Chunking import Chunker
from Retrieval import Retriever
from Agent import Agent

load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(exist_ok=True)

EXT_TO_MIME = {
    ".txt": MimeType.TXT,
    ".csv": MimeType.CSV,
    ".xlsx": MimeType.XLSX,
}

app = FastAPI(title="Finance Agent")

state: dict = {"agent": None, "files": []}


def _rebuild_agent() -> None:
    if not state["files"]:
        state["agent"] = None
        return
    paths = [f["path"] for f in state["files"]]
    mimes = [f["mime"] for f in state["files"]]
    retriever = Retriever(paths, mimes, topk=1, ingestor=Ingest, chunker=Chunker)
    state["agent"] = Agent(retriever)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/files")
def list_files():
    return {"files": [f["name"] for f in state["files"]]}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        if not f.filename:
            continue
        ext = pathlib.Path(f.filename).suffix.lower()
        if ext not in EXT_TO_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Allowed: .txt, .csv, .xlsx",
            )
        dest = UPLOAD_DIR / f.filename
        dest.write_bytes(await f.read())
        if not any(x["path"] == str(dest) for x in state["files"]):
            state["files"].append(
                {"path": str(dest), "mime": EXT_TO_MIME[ext], "name": f.filename}
            )
        saved.append(f.filename)

    _rebuild_agent()
    return {"uploaded": saved, "files": [f["name"] for f in state["files"]]}


@app.post("/reset")
def reset():
    for f in state["files"]:
        try:
            os.remove(f["path"])
        except OSError:
            pass
    state["files"] = []
    state["agent"] = None
    return {"ok": True}


def _semantic_citation(c) -> dict:
    chunk = c.chunk
    return {
        "source_file": os.path.basename(chunk.filename or ""),
        "title": chunk.title or "",
        "author": chunk.author or "",
        "participants": chunk.participants or [],
        "date": chunk.date or "",
        "content": chunk.content,
        "score": float(c.score),
    }


def _numeric_citation(n) -> dict:
    df = n.table.reset_index() if getattr(n.table, "index", None) is not None else n.table
    columns = [str(c) for c in df.columns]
    rows = [
        {str(k): (None if _is_nan(v) else v) for k, v in rec.items()}
        for rec in df.to_dict(orient="records")
    ]
    return {
        "source_file": os.path.basename(n.source_file or ""),
        "sql": n.sql,
        "columns": columns,
        "rows": rows,
    }


def _is_nan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


@app.post("/query")
def query(q: str = Form(...)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    if state["agent"] is None:
        return {
            "answer": (
                "Cannot be answered: no sources have been uploaded. "
                "Upload at least one .txt, .csv, or .xlsx file to provide evidence."
            ),
            "citations": {"semantic": [], "numeric": []},
        }

    resp = state["agent"].generate_response(q)

    return {
        "answer": resp.response,
        "citations": {
            "semantic": [_semantic_citation(c) for c in resp.semantic_citations],
            "numeric": [
                _numeric_citation(n) for n in resp.numeric_citations
                if not n.table.empty
            ],
        },
    }
