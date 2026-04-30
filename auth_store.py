"""SQLite-backed user store with bcrypt password hashing."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from passlib.context import CryptContext

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def create_user(db_path: Path, username: str, password: str) -> dict:
    username = username.strip()
    if len(username) < 2:
        raise ValueError("Username must be at least 2 characters.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    ph = hash_password(password)
    now = _utc_now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, ph, now),
        )
        uid = int(cur.lastrowid)
    return {"id": uid, "username": username}


def get_user_by_id(db_path: Path, user_id: int) -> Optional[dict]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


def get_user_by_username(db_path: Path, username: str) -> Optional[dict]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
    }


def authenticate(db_path: Path, username: str, password: str) -> Optional[dict]:
    user = get_user_by_username(db_path, username)
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}
