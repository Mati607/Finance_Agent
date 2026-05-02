import mimetypes
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware

from Agent import Agent
from Chunking import Chunker
from Ingestion import Ingest, MimeType
from Retrieval import Retriever

import auth_store

load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
USER_DB = DATA_DIR / "users.db"

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-only-insecure-secret-change-in-production")
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "true").lower() in (
    "1",
    "true",
    "yes",
)

auth_store.init_db(USER_DB)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXT_TO_MIME = {
    ".txt": MimeType.TXT,
    ".csv": MimeType.CSV,
    ".xlsx": MimeType.XLSX,
}

app = FastAPI(title="Finance Agent")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="session",
    max_age=14 * 24 * 3600,
    same_site="lax",
    https_only=False,
)

# Per-user agent state: user_id -> {"agent": Agent | None, "files": [...]}
user_states: dict[int, dict] = {}

@dataclass
class RateLimiter:
    limit: int
    window_s: int
    hits: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_s
        bucket = self.hits.get(key, [])
        bucket = [t for t in bucket if t >= cutoff]
        if len(bucket) >= self.limit:
            self.hits[key] = bucket
            return False
        bucket.append(now)
        self.hits[key] = bucket
        return True


auth_rl = RateLimiter(limit=20, window_s=60)
query_rl = RateLimiter(limit=30, window_s=60)


def _client_ip(request: Request) -> str:
    return getattr(request.client, "host", None) or "unknown"


def rate_limit(kind: str):
    def dep(request: Request):
        ip = _client_ip(request)
        key = f"{kind}:{ip}"
        rl = auth_rl if kind == "auth" else query_rl
        if not rl.allow(key):
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    return Depends(dep)


def require_admin(user: dict) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return user


def _user_upload_dir(user_id: int) -> pathlib.Path:
    d = UPLOAD_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _files_payload(st: dict) -> list[dict]:
    out: list[dict] = []
    for f in st["files"]:
        p = pathlib.Path(f["path"])
        try:
            sz = int(p.stat().st_size)
        except OSError:
            sz = None
        out.append({"name": f["name"], "size_bytes": sz})
    return out


def _safe_upload_file_path(user_id: int, raw_name: str) -> pathlib.Path:
    name = pathlib.Path(raw_name).name.strip()
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    root = _user_upload_dir(user_id).resolve()
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found.")
    return path


def get_user_state(user_id: int) -> dict:
    if user_id not in user_states:
        user_states[user_id] = {"agent": None, "files": []}
    return user_states[user_id]


def _rebuild_agent(user_id: int) -> None:
    st = get_user_state(user_id)
    if not st["files"]:
        st["agent"] = None
        return
    paths = [f["path"] for f in st["files"]]
    mimes = [f["mime"] for f in st["files"]]
    retriever = Retriever(
        paths,
        mimes,
        ingestor=Ingest,
        chunker=Chunker,
        topk_retrieve=20,
        topk_final=3,
    )
    st["agent"] = Agent(retriever)


def current_user(request: Request) -> dict:
    uid = request.session.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth_store.get_user_by_id(USER_DB, int(uid))
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


UserDep = Annotated[dict, Depends(current_user)]


@app.post("/auth/register")
async def register(
    request: Request,
    _rl=rate_limit("auth"),
    username: str = Form(...),
    password: str = Form(...),
):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registration is disabled.")
    try:
        user = auth_store.create_user(USER_DB, username, password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="That username is already taken.")
    request.session["user_id"] = user["id"]
    return {"ok": True, "username": user["username"]}


@app.post("/auth/login")
async def login(
    request: Request,
    _rl=rate_limit("auth"),
    username: str = Form(...),
    password: str = Form(...),
):
    user = auth_store.authenticate(USER_DB, username, password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    request.session["user_id"] = user["id"]
    return {"ok": True, "username": user["username"]}


@app.post("/auth/logout")
def logout(request: Request, _rl=rate_limit("auth")):
    request.session.clear()
    return {"ok": True}


@app.get("/auth/config")
def auth_config():
    return {"allow_registration": ALLOW_REGISTRATION}


@app.get("/auth/me")
def me(user: UserDep):
    return {"id": user["id"], "username": user["username"], "role": user.get("role", "user")}


@app.post("/auth/admin/set-password")
def admin_set_password(
    user: UserDep,
    _rl=rate_limit("auth"),
    username: str = Form(...),
    new_password: str = Form(...),
):
    require_admin(user)
    try:
        ok = auth_store.set_password(USER_DB, username, new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/files")
def list_files(user: UserDep):
    st = get_user_state(user["id"])
    return {"files": _files_payload(st)}


@app.get("/files/download/{filename:path}")
def download_file(user: UserDep, filename: str):
    uid = user["id"]
    st = get_user_state(uid)
    path = _safe_upload_file_path(uid, filename)
    allowed = {pathlib.Path(x["path"]).resolve() for x in st["files"]}
    if path.resolve() not in allowed:
        raise HTTPException(status_code=404, detail="File not found.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    media_type, _ = mimetypes.guess_type(path.name)
    if not media_type:
        media_type = "application/octet-stream"

    return FileResponse(
        path,
        filename=path.name,
        media_type=media_type,
    )


@app.post("/upload")
async def upload(user: UserDep, files: list[UploadFile] = File(...)):
    uid = user["id"]
    udir = _user_upload_dir(uid)
    st = get_user_state(uid)
    saved = []
    for f in files:
        if not f.filename:
            continue
        safe_name = pathlib.Path(f.filename).name.strip()
        if not safe_name:
            continue
        ext = pathlib.Path(safe_name).suffix.lower()
        if ext not in EXT_TO_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Allowed: .txt, .csv, .xlsx",
            )
        dest = udir / safe_name
        dest.write_bytes(await f.read())
        if not any(x["path"] == str(dest) for x in st["files"]):
            st["files"].append(
                {"path": str(dest), "mime": EXT_TO_MIME[ext], "name": safe_name}
            )
        saved.append(safe_name)

    _rebuild_agent(uid)
    return {"uploaded": saved, "files": _files_payload(st)}


@app.post("/reset")
def reset(user: UserDep):
    uid = user["id"]
    st = get_user_state(uid)
    for f in st["files"]:
        try:
            os.remove(f["path"])
        except OSError:
            pass
    st["files"] = []
    st["agent"] = None
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
def query(user: UserDep, _rl=rate_limit("query"), q: str = Form(...)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    st = get_user_state(user["id"])
    if st["agent"] is None:
        return {
            "answer": (
                "Cannot be answered: no sources have been uploaded. "
                "Upload at least one .txt, .csv, or .xlsx file to provide evidence."
            ),
            "citations": {"semantic": [], "numeric": []},
        }

    resp = st["agent"].generate_response(q)

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
