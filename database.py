import sqlite3
import os
from datetime import datetime

DB_NAME = "chat_logs.db"

def init_db():
    """
    Initializes the SQLite database and creates the chat_logs and thread_settings tables if they don't exist.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_phone TEXT NOT NULL,
            user_message TEXT,
            bot_response TEXT,
            provider TEXT,
            model TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_settings (
            thread_id TEXT PRIMARY KEY,
            auto_reply INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active'
        )
    """)
    conn.commit()
    conn.close()

def log_chat(sender_phone: str, user_message: str, bot_response: str, provider: str, model: str):
    """
    Inserts a new chat transaction log into the database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO chat_logs (sender_phone, user_message, bot_response, provider, model, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (sender_phone, user_message, bot_response, provider, model, now))
    conn.commit()
    conn.close()

def get_recent_logs(limit: int = 50):
    """
    Retrieves the most recent chat logs for review.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sender_phone, user_message, bot_response, provider, model, timestamp 
        FROM chat_logs 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_thread_settings(thread_id: str) -> dict:
    """
    Retrieves auto-reply toggle state and status for a specific chat thread.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT auto_reply, status FROM thread_settings WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"auto_reply": bool(row[0]), "status": row[1]}
    return {"auto_reply": True, "status": "active"}

def set_thread_settings(thread_id: str, auto_reply: bool, status: str):
    """
    Updates or inserts settings for a specific chat thread.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    auto_reply_val = 1 if auto_reply else 0
    cursor.execute("""
        INSERT INTO thread_settings (thread_id, auto_reply, status)
        VALUES (?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET auto_reply = excluded.auto_reply, status = excluded.status
    """, (thread_id, auto_reply_val, status))
    conn.commit()
    conn.close()

def get_active_threads() -> list:
    """
    Retrieves all conversation threads that have chatted, along with their settings.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cl.thread_id, COALESCE(ts.auto_reply, 1), COALESCE(ts.status, 'active')
        FROM (SELECT DISTINCT sender_phone AS thread_id FROM chat_logs) cl
        LEFT JOIN thread_settings ts ON cl.thread_id = ts.thread_id
        ORDER BY cl.thread_id DESC
    """)
    threads = cursor.fetchall()
    conn.close()
    return threads

