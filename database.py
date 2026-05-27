import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_FILE = Path(__file__).resolve().parent / "blogs.db"

def get_db_connection():
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE,
                title TEXT,
                final_markdown TEXT,
                evidence_json TEXT,
                plan_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

def save_blog(slug: str, title: str, markdown: str, evidence: List[Dict[str, Any]], plan: Optional[Dict[str, Any]]) -> int:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        evidence_str = json.dumps(evidence)
        plan_str = json.dumps(plan) if plan else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO blogs (slug, title, final_markdown, evidence_json, plan_json)
            VALUES (?, ?, ?, ?, ?)
        """, (slug, title, markdown, evidence_str, plan_str))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def list_blogs() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, slug, title, created_at FROM blogs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_blog(blog_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, slug, title, final_markdown, evidence_json, plan_json, created_at FROM blogs WHERE id = ?", (blog_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        blog_dict = dict(row)
        # Parse JSON fields back to objects
        try:
            blog_dict["evidence"] = json.loads(blog_dict["evidence_json"]) if blog_dict["evidence_json"] else []
        except Exception:
            blog_dict["evidence"] = []
            
        try:
            blog_dict["plan"] = json.loads(blog_dict["plan_json"]) if blog_dict["plan_json"] else None
        except Exception:
            blog_dict["plan"] = None
            
        return blog_dict
    finally:
        conn.close()

def delete_blog(blog_id: int) -> bool:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM blogs WHERE id = ?", (blog_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

# Auto-initialize database on module load
init_db()
