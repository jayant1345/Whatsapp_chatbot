import os
import logging
import requests
import sqlite3
from collections import deque
from typing import Annotated, TypedDict
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, Request, Query, Response, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# LangChain / LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage

# Custom database logging and RAG helper imports
from database import (
    init_db, 
    log_chat, 
    get_recent_logs, 
    get_thread_settings, 
    set_thread_settings, 
    get_active_threads
)
from rag_helper import index_knowledge_base, query_knowledge_base, get_indexed_files

# Load environment variables
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

# Setup system console and file logging mechanism
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_file_path = os.path.join(DATA_DIR, "server_logs.txt")

file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Apply logging handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info("Server started. Logging system initialized.")

# Verify required keys
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")

app = FastAPI(title="WhatsApp Chatbot Dashboard")

# Recently processed WhatsApp message IDs, to ignore duplicate webhook redeliveries from Meta
_recent_message_ids = deque(maxlen=1000)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    logger.info("Initializing chat logging database...")
    init_db()

# Define the state shape
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Define LLM selector node
def call_model(state: AgentState):
    messages = state["messages"]
    user_query = messages[-1].content
    
    # Reload environment variables inside node for real-time dashboard config changes
    load_dotenv()
    provider = os.getenv("AI_PROVIDER", "anthropic").lower()
    model_name = os.getenv("SELECTED_MODEL")
    
    # Retrieve semantic context from pgvector database
    try:
        context = query_knowledge_base(user_query)
        if context:
            logger.info(f"Retrieved context from pgvector: {context[:100]}...")
    except Exception as e:
        logger.error(f"RAG pgvector retrieval failed: {e}")
        context = ""
        
    system_content = (
        "You are a professional customer service assistant for our client JK Data Lab. "
        "Keep your answers concise, accurate, and under 3 sentences. "
        "This is a WhatsApp chat, so format replies the way a helpful human would text: "
        "use 1-2 relevant emoji per message (never more, never one per sentence), and put list items "
        "or multi-part answers on separate lines instead of one dense paragraph. Don't force an emoji "
        "into a reply where it wouldn't naturally fit."
    )
    if context:
        system_content += f"\n\nUse the following verified context from the company database to answer the query:\n{context}\nIf the answer is not found in this context, politely say that you don't know."
        
    system_prompt = SystemMessage(content=system_content)
    full_messages = [system_prompt] + messages
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.5,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.5,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Invalid AI_PROVIDER: {provider}")
        
    response = llm.invoke(full_messages)
    return {"messages": [response]}

# Compile the LangGraph with SQLite checkpointing for persistent memory
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Initialize SqliteSaver manually for 24/7 local thread memory
db_conn = sqlite3.connect(os.path.join(DATA_DIR, "checkpoints.sqlite"), check_same_thread=False)
checkpointer = SqliteSaver(db_conn)
checkpointer.setup()
chatbot_graph = workflow.compile(checkpointer=checkpointer)

# ---------------------------------------------------------
# 1. API Endpoints for Web Dashboard
# ---------------------------------------------------------

@app.get("/api/config")
def get_config():
    """
    Returns current configuration variables to the dashboard.
    """
    load_dotenv()
    return {
        "ai_provider": os.getenv("AI_PROVIDER", "anthropic"),
        "selected_model": os.getenv("SELECTED_MODEL", ""),
        "meta_phone_id": os.getenv("META_PHONE_NUMBER_ID", ""),
        "has_anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_meta_token": bool(os.getenv("META_ACCESS_TOKEN")),
        
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": os.getenv("POSTGRES_PORT", "5432"),
        "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
        "postgres_db": os.getenv("POSTGRES_DB", "postgres")
    }

@app.post("/api/config")
async def update_config(request: Request):
    """
    Updates configuration variables and saves them directly to the .env file.
    """
    data = await request.json()
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    
    try:
        if "ai_provider" in data:
            set_key(dotenv_path, "AI_PROVIDER", data["ai_provider"])
        if "selected_model" in data:
            set_key(dotenv_path, "SELECTED_MODEL", data["selected_model"])
        if "anthropic_key" in data and data["anthropic_key"]:
            set_key(dotenv_path, "ANTHROPIC_API_KEY", data["anthropic_key"])
        if "openrouter_key" in data and data["openrouter_key"]:
            set_key(dotenv_path, "OPENROUTER_API_KEY", data["openrouter_key"])
        if "openai_key" in data and data["openai_key"]:
            set_key(dotenv_path, "OPENAI_API_KEY", data["openai_key"])
        if "meta_token" in data and data["meta_token"]:
            set_key(dotenv_path, "META_ACCESS_TOKEN", data["meta_token"])
        if "meta_phone_id" in data:
            set_key(dotenv_path, "META_PHONE_NUMBER_ID", data["meta_phone_id"])
            
        if "postgres_host" in data:
            set_key(dotenv_path, "POSTGRES_HOST", data["postgres_host"])
        if "postgres_port" in data:
            set_key(dotenv_path, "POSTGRES_PORT", data["postgres_port"])
        if "postgres_user" in data:
            set_key(dotenv_path, "POSTGRES_USER", data["postgres_user"])
        if "postgres_password" in data and data["postgres_password"]:
            set_key(dotenv_path, "POSTGRES_PASSWORD", data["postgres_password"])
        if "postgres_db" in data:
            set_key(dotenv_path, "POSTGRES_DB", data["postgres_db"])
            
        load_dotenv(override=True)
        logger.info("System configuration settings updated successfully.")
        return {"status": "success", "message": "Configuration updated successfully."}
    except Exception as e:
        logger.error(f"Failed to save system configurations: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/logs")
def get_logs():
    """
    Returns recent chat logs from the SQLite database.
    """
    logs = get_recent_logs(50)
    formatted = []
    for log in logs:
        formatted.append({
            "sender": log[0],
            "message": log[1],
            "response": log[2],
            "provider": log[3],
            "model": log[4],
            "timestamp": log[5]
        })
    return formatted

@app.post("/api/chat")
async def local_chat_test(request: Request):
    """
    Provides a chatbot testing interface directly inside the web dashboard.
    """
    data = await request.json()
    message_text = data.get("message", "")
    
    # Use a generic thread ID for dashboard chat tests
    thread_id = "dashboard_test_user"
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        logger.info(f"Sandbox user sent query: '{message_text}'")
        inputs = {"messages": [HumanMessage(content=message_text)]}
        output_state = chatbot_graph.invoke(inputs, config=config)
        bot_response = output_state["messages"][-1].content
        
        # Log this interaction to the local SQLite logs database
        load_dotenv()
        log_chat(
            thread_id, 
            message_text, 
            bot_response, 
            os.getenv("AI_PROVIDER", "anthropic"), 
            os.getenv("SELECTED_MODEL", "")
        )
        logger.info(f"Sandbox bot generated response: '{bot_response}'")
        return {"response": bot_response}
    except Exception as e:
        logger.exception("Error in local web chat:")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/knowledge/status")
def get_knowledge_status():
    """
    Checks the status of indexed knowledge base files in pgvector.
    """
    files = get_indexed_files()
    return {
        "status": "indexed" if files else "empty",
        "files": files
    }

@app.post("/api/knowledge/index")
def sync_knowledge_index():
    """
    Indexes all files inside the knowledge directory into pgvector.
    """
    try:
        logger.info("Initializing pgvector semantic indexing process...")
        chunks = index_knowledge_base()
        logger.info(f"Indexing complete. Successfully loaded {chunks} semantic chunks into pgvector.")
        return {"status": "success", "message": f"Successfully indexed {chunks} semantic chunks in pgvector."}
    except Exception as e:
        logger.error(f"Error during pgvector semantic indexing: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...)):
    """
    Saves uploaded knowledge text document directly into the knowledge folder.
    """
    try:
        if not file.filename.endswith(".txt"):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Only .txt files are supported."})
            
        knowledge_dir = os.path.join(DATA_DIR, "knowledge")
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)
            
        file_path = os.path.join(knowledge_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        logger.info(f"File '{file.filename}' uploaded successfully to knowledge folder.")
        return {"status": "success", "message": f"Successfully uploaded {file.filename}."}
    except Exception as e:
        logger.error(f"Failed to upload document file: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/system-logs")
def get_system_logs():
    """
    Reads the server console logs file and returns the last 50 lines.
    """
    if not os.path.exists(log_file_path):
        return {"logs": ["Console logger is starting up. File server_logs.txt not found yet."]}
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-50:]]}
    except Exception as e:
        return {"logs": [f"Error reading system logs: {e}"]}

# ---------------------------------------------------------
# Admin Chat Inbox endpoints (Human-in-the-loop WhatsApp reports)
# ---------------------------------------------------------

@app.get("/api/inbox/threads")
def get_inbox_threads():
    """
    Returns list of all distinct conversation threads, their bot settings, and escalation status.
    """
    threads = get_active_threads()
    formatted = []
    for thread in threads:
        formatted.append({
            "thread_id": thread[0],
            "auto_reply": bool(thread[1]),
            "status": thread[2]
        })
    return formatted

@app.get("/api/inbox/thread/{thread_id}")
def get_thread_chat_history(thread_id: str):
    """
    Returns chat message history and settings for a specific phone number.
    """
    conn = sqlite3.connect(os.path.join(DATA_DIR, "chat_logs.db"))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_message, bot_response, provider, model, timestamp 
        FROM chat_logs 
        WHERE sender_phone = ? 
        ORDER BY id ASC
    """, (thread_id,))
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        messages.append({
            "user_message": row[0],
            "bot_response": row[1],
            "provider": row[2],
            "model": row[3],
            "timestamp": row[4]
        })
        
    settings = get_thread_settings(thread_id)
    return {
        "thread_id": thread_id,
        "auto_reply": settings["auto_reply"],
        "status": settings["status"],
        "messages": messages
    }

@app.post("/api/inbox/thread/{thread_id}/settings")
async def update_thread_settings_endpoint(thread_id: str, request: Request):
    """
    Toggles auto reply status or resets escalation.
    """
    data = await request.json()
    auto_reply = data.get("auto_reply", True)
    status_val = data.get("status", "active")
    
    set_thread_settings(thread_id, auto_reply, status_val)
    logger.info(f"Thread '{thread_id}' updated: AutoReply={auto_reply}, Status={status_val}")
    return {"status": "success"}

@app.post("/api/inbox/thread/{thread_id}/send")
async def send_manual_message_endpoint(thread_id: str, request: Request):
    """
    Sends a manual text message to the customer via WhatsApp and updates settings to manual mode.
    """
    data = await request.json()
    message_text = data.get("message", "")
    
    if not message_text:
        return JSONResponse(status_code=400, content={"error": "Message text cannot be empty."})
        
    logger.info(f"Admin sending manual message to {thread_id}: '{message_text}'")
    
    # 1. Send via WhatsApp Graph API
    send_whatsapp_message(thread_id, message_text)
    
    # 2. Log manual message in chat_logs (user_message contains label, bot_response has the actual text)
    log_chat(thread_id, "[Manual Admin message]", message_text, "admin", "manual")
    
    # 3. Update thread settings: Auto reply remains paused (since human took over) and reset escalation to active
    set_thread_settings(thread_id, auto_reply=False, status="active")
    
    return {"status": "success"}

@app.get("/api/global-auto-reply")
def get_global_auto_reply():
    """
    Returns global bot auto-reply status from environment.
    """
    load_dotenv()
    return {"global_auto_reply": os.getenv("GLOBAL_BOT_AUTO_REPLY", "True") == "True"}

@app.post("/api/global-auto-reply")
async def toggle_global_auto_reply(request: Request):
    """
    Toggles global bot auto-reply status.
    """
    data = await request.json()
    status_val = "True" if data.get("global_auto_reply", True) else "False"
    
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        set_key(dotenv_path, "GLOBAL_BOT_AUTO_REPLY", status_val)
        load_dotenv(override=True)
        logger.info(f"Global Bot Auto-Reply set to: {status_val}")
        return {"status": "success", "global_auto_reply": status_val == "True"}
    except Exception as e:
        logger.error(f"Failed to toggle global auto-reply: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ---------------------------------------------------------
# 2. Web Interface (Premium UI Design based on Google Stitch)
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JK Data Lab Chatbot Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                /* Light Greenish Tonal Design System */
                --bg-primary: #070d0b;
                --bg-secondary: #0b1310;
                --surface-container: #111d19;
                --surface-bright: #1c2e27;
                --primary: #a7f3d0;
                --on-primary: #042f1a;
                --primary-container: #064e3b;
                --secondary: #ffb77d;
                --tertiary: #34d399;
                --error: #ef4444;
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --outline: #1e352b;
                --outline-variant: #2d4c3e;
                --glow-shadow: 0 0 20px rgba(167, 243, 208, 0.25);
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Inter', sans-serif;
            }

            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                display: flex;
                flex-direction: column;
                height: 100vh;
                overflow: hidden;
            }

            header {
                background-color: var(--bg-secondary);
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--outline);
            }

            .logo-section h1 {
                font-size: 1.4rem;
                font-weight: 800;
                letter-spacing: 0.5px;
                background: linear-gradient(45deg, var(--tertiary), var(--primary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .logo-section span {
                font-size: 0.75rem;
                color: var(--text-secondary);
                font-weight: 400;
            }

            .header-actions {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .status-container {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(78, 222, 163, 0.08);
                border: 1px solid rgba(78, 222, 163, 0.2);
                padding: 0.35rem 0.8rem;
                border-radius: 50px;
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--tertiary);
            }

            .status-indicator {
                width: 8px;
                height: 8px;
                background-color: var(--tertiary);
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(78, 222, 163, 0.6);
                animation: pulse 2s infinite;
            }

            /* Main Header Global Switch Styles */
            .global-switch-container {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: var(--surface-container);
                border: 1px solid var(--outline);
                padding: 0.4rem 1rem;
                border-radius: 6px;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .switch {
                position: relative;
                display: inline-block;
                width: 44px;
                height: 22px;
            }

            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }

            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: var(--outline-variant);
                transition: .4s;
                border-radius: 34px;
            }

            .slider:before {
                position: absolute;
                content: "";
                height: 16px;
                width: 16px;
                left: 3px;
                bottom: 3px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }

            input:checked + .slider {
                background-color: var(--tertiary);
                box-shadow: 0 0 10px rgba(52, 211, 153, 0.5);
            }

            input:checked + .slider:before {
                transform: translateX(22px);
            }

            .btn-settings {
                background: var(--surface-container);
                border: 1px solid var(--outline);
                color: var(--primary);
                padding: 0.55rem 1.2rem;
                border-radius: 6px;
                font-size: 0.85rem;
                font-weight: 600;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                transition: all 0.3s;
            }

            .btn-settings:hover {
                border-color: var(--primary);
                box-shadow: var(--glow-shadow);
                color: var(--text-primary);
            }

            main {
                display: flex;
                justify-content: center;
                align-items: center;
                flex: 1;
                padding: 2rem;
                overflow: hidden;
            }

            .panel-card {
                background-color: var(--surface-container);
                border: 1px solid var(--outline);
                border-radius: 8px;
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }

            /* Main Chat Simulator sizing */
            .chat-simulator-container {
                width: 700px;
                height: 100%;
                max-height: 750px;
            }

            /* Main Homepage Inbox sizing */
            .inbox-container {
                width: 900px;
                height: 100%;
                max-height: 750px;
            }

            .panel-card h2 {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--outline-variant);
                padding-bottom: 0.5rem;
                color: var(--text-primary);
            }

            /* Chat Sandbox Styles */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 0.8rem;
                background-color: var(--bg-primary);
                border-radius: 6px;
                border: 1px solid var(--outline);
                margin-bottom: 0.8rem;
            }

            .msg {
                max-width: 80%;
                padding: 0.7rem 1rem;
                border-radius: 8px;
                font-size: 0.85rem;
                line-height: 1.4;
            }

            .msg.user {
                align-self: flex-end;
                background-color: var(--primary-container);
                color: var(--text-primary);
                border: 1px solid var(--outline);
                border-bottom-right-radius: 2px;
            }

            .msg.bot {
                align-self: flex-start;
                background-color: var(--surface-container);
                border: 1px solid var(--outline);
                border-bottom-left-radius: 2px;
            }

            .chat-input-area {
                display: flex;
                gap: 0.5rem;
            }

            .chat-input-area input {
                flex: 1;
                background-color: var(--bg-primary);
                border: 1px solid var(--outline);
                padding: 0.6rem 0.8rem;
                border-radius: 6px;
                color: var(--text-primary);
                outline: none;
                font-size: 0.85rem;
                transition: border-color 0.3s;
            }

            .chat-input-area input:focus {
                border-color: var(--primary);
            }

            .btn {
                background: var(--primary);
                color: var(--on-primary);
                border: none;
                padding: 0.6rem 1.2rem;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.85rem;
                cursor: pointer;
                transition: filter 0.3s;
            }

            .btn:hover {
                filter: brightness(1.15);
            }

            .btn-secondary {
                background: transparent;
                color: var(--primary);
                border: 1px solid var(--outline);
                padding: 0.55rem 1.2rem;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.85rem;
                cursor: pointer;
                transition: all 0.3s;
            }

            .btn-secondary:hover {
                border-color: var(--primary);
                background: rgba(167, 243, 208, 0.05);
            }

            .badge {
                font-size: 0.7rem;
                background-color: var(--outline-variant);
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                font-weight: 600;
                display: inline-block;
            }
            
            .badge.anthropic { color: #ffb77d; background: rgba(255, 183, 125, 0.08); border: 1px solid rgba(255, 183, 125, 0.2); }
            .badge.openrouter { color: var(--primary); background: rgba(173, 198, 255, 0.08); border: 1px solid rgba(173, 198, 255, 0.2); }

            /* Settings Console Overlay Modal */
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(7, 13, 11, 0.85);
                backdrop-filter: blur(8px);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 100;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s ease;
            }

            .modal-overlay.active {
                opacity: 1;
                pointer-events: auto;
            }

            .modal-content {
                background-color: var(--surface-container);
                border: 1px solid var(--outline);
                border-radius: 12px;
                width: 950px;
                max-width: 95%;
                padding: 1.8rem;
                box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6);
                transform: scale(0.95);
                transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
                height: 85vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .modal-overlay.active .modal-content {
                transform: scale(1);
            }

            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--outline-variant);
                padding-bottom: 0.75rem;
                margin-bottom: 1rem;
            }

            .modal-header h2 {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--primary);
            }

            .close-btn {
                background: none;
                border: none;
                color: var(--text-secondary);
                font-size: 1.5rem;
                cursor: pointer;
                transition: color 0.3s;
            }

            .close-btn:hover {
                color: var(--error);
            }

            /* Settings Tab Styling */
            .tabs-nav {
                display: flex;
                gap: 0.5rem;
                border-bottom: 1px solid var(--outline);
                padding-bottom: 0.5rem;
                margin-bottom: 1.2rem;
            }

            .tab-btn {
                background: transparent;
                border: 1px solid transparent;
                color: var(--text-secondary);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.85rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }

            .tab-btn.active {
                background: var(--bg-primary);
                border-color: var(--outline);
                color: var(--primary);
                box-shadow: var(--glow-shadow);
            }

            .tab-btn:hover:not(.active) {
                color: var(--text-primary);
                background: rgba(167, 243, 208, 0.05);
            }

            .tab-content {
                display: none;
                flex: 1;
                overflow-y: auto;
                padding-right: 0.25rem;
            }

            .tab-content.active {
                display: flex;
                flex-direction: column;
            }

            /* Configuration Form layouts */
            form {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }

            .form-span-full {
                grid-column: 1 / span 2;
            }

            .form-section-title {
                grid-column: 1 / span 2;
                font-size: 0.8rem;
                font-weight: 700;
                color: var(--primary);
                margin-top: 0.5rem;
                border-bottom: 1px dashed var(--outline);
                padding-bottom: 0.25rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .input-group {
                display: flex;
                flex-direction: column;
                gap: 0.3rem;
            }

            .input-group label {
                font-size: 0.75rem;
                color: var(--text-secondary);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            input, select {
                background-color: var(--bg-primary);
                border: 1px solid var(--outline);
                padding: 0.55rem 0.8rem;
                border-radius: 6px;
                color: var(--text-primary);
                outline: none;
                font-size: 0.85rem;
                transition: border-color 0.3s;
            }

            input:focus, select:focus {
                border-color: var(--primary);
            }

            /* Logs layout inside Tab */
            .logs-container {
                flex: 1;
                overflow-y: auto;
            }

            .log-item {
                background-color: var(--bg-primary);
                border: 1px solid var(--outline);
                padding: 0.8rem 1rem;
                border-radius: 6px;
                margin-bottom: 0.6rem;
                font-size: 0.85rem;
            }

            .log-header {
                display: flex;
                justify-content: space-between;
                color: var(--text-secondary);
                margin-bottom: 0.4rem;
                font-size: 0.75rem;
            }

            /* Console logs terminal inside Tab */
            #sysLogsBox {
                background-color: #020408;
                border: 1px solid var(--outline);
                border-radius: 6px;
                padding: 1rem;
                font-family: monospace;
                font-size: 0.75rem;
                color: #a7f3d0;
                flex: 1;
                overflow-y: auto;
                white-space: pre-wrap;
                line-height: 1.4;
            }

            /* Admin Inbox Specific Styles */
            .inbox-layout {
                display: grid;
                grid-template-columns: 240px 1fr;
                gap: 1rem;
                flex: 1;
                overflow: hidden;
            }

            .thread-list {
                border-right: 1px solid var(--outline);
                overflow-y: auto;
                padding-right: 0.5rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            .thread-card {
                background-color: var(--bg-primary);
                border: 1px solid var(--outline);
                padding: 0.75rem 1rem;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s;
                text-align: left;
            }

            .thread-card:hover {
                border-color: var(--primary);
            }

            .thread-card.active-select {
                background-color: var(--outline);
                border-color: var(--primary);
            }

            .thread-phone {
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 0.25rem;
            }

            .thread-status-line {
                display: flex;
                gap: 0.3rem;
                font-size: 0.7rem;
            }

            .badge-inbox {
                font-size: 0.65rem;
                padding: 0.1rem 0.35rem;
                border-radius: 4px;
                font-weight: 700;
            }

            .badge-inbox.bot-on { background: rgba(52, 211, 153, 0.15); color: var(--tertiary); }
            .badge-inbox.bot-off { background: rgba(156, 163, 175, 0.15); color: var(--text-secondary); }
            .badge-inbox.status-esc { background: rgba(239, 68, 68, 0.15); color: var(--error); animation: pulse 1.5s infinite; }

            .chat-window {
                display: flex;
                flex-direction: column;
                overflow: hidden;
                background-color: var(--bg-primary);
                border: 1px solid var(--outline);
                border-radius: 8px;
                padding: 1rem;
            }

            .chat-window-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--outline);
                padding-bottom: 0.6rem;
                margin-bottom: 0.8rem;
            }

            .chat-window-messages {
                flex: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 0.8rem;
                padding: 0.5rem;
                margin-bottom: 0.8rem;
            }

            .chat-bubble {
                max-width: 75%;
                padding: 0.6rem 0.9rem;
                border-radius: 8px;
                font-size: 0.8rem;
                line-height: 1.4;
            }

            .chat-bubble.user {
                align-self: flex-start;
                background-color: var(--bg-secondary);
                border: 1px solid var(--outline);
                border-bottom-left-radius: 2px;
            }

            .chat-bubble.bot {
                align-self: flex-end;
                background-color: var(--primary-container);
                color: var(--text-primary);
                border: 1px solid var(--outline);
                border-bottom-right-radius: 2px;
            }

            .chat-bubble.admin {
                align-self: flex-end;
                background-color: var(--outline-variant);
                color: var(--text-primary);
                border: 1px solid var(--outline);
                border-bottom-right-radius: 2px;
            }

            @keyframes pulse {
                0% { transform: scale(0.95); opacity: 0.5; }
                50% { transform: scale(1.05); opacity: 1; }
                100% { transform: scale(0.95); opacity: 0.5; }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo-section">
                <h1>JK Data Lab</h1>
                <span>WhatsApp AI Chatbot Dashboard v1.0</span>
            </div>
            <div class="header-actions">
                <!-- Global Auto Reply Switch (directly in header) -->
                <div class="global-switch-container">
                    <span id="switchStatusText" style="color: var(--primary); margin-right: 0.2rem;">🤖 Bot Active</span>
                    <label class="switch">
                        <input type="checkbox" id="globalBotToggle" onchange="handleGlobalToggle(this)" checked>
                        <span class="slider"></span>
                    </label>
                </div>
                
                <div class="status-container">
                    <div class="status-indicator"></div>
                    <span>Server Status: Live</span>
                </div>
                <button class="btn-settings" onclick="openSettings()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    Settings & Control Panel
                </button>
            </div>
        </header>

        <main>
            <!-- WhatsApp Support Inbox (Main Homepage Interface) -->
            <div class="panel-card inbox-container">
                <h2>💬 WhatsApp Support Inbox</h2>
                <div class="inbox-layout">
                    <!-- Left pane -->
                    <div class="thread-list" id="inboxThreadList">
                        <span style="color:var(--text-secondary); text-align:center; margin-top:2rem; font-size:0.8rem;">No active WhatsApp conversations yet.</span>
                    </div>

                    <!-- Right pane -->
                    <div class="chat-window" id="inboxChatWindow">
                        <div style="color:var(--text-secondary); text-align:center; margin-top:6rem; font-size:0.85rem;">
                            Select a phone number from the left to view history, toggle auto-reply, or reply manually over WhatsApp.
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Configuration Settings Modal Overlay -->
        <div class="modal-overlay" id="settingsModal" onclick="closeSettings(event)">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h2>JK Data Lab Chatbot Control Panel</h2>
                    <button class="close-btn" onclick="closeSettings()">&times;</button>
                </div>
                
                <!-- Modal Tabs Navigation -->
                <div class="tabs-nav">
                    <button class="tab-btn active" onclick="switchTab('tab-sandbox')">🧪 Chat Sandbox (Local Test)</button>
                    <button class="tab-btn" onclick="switchTab('tab-config')">⚙ System Config</button>
                    <button class="tab-btn" onclick="switchTab('tab-knowledge')">📚 Knowledge Base (RAG)</button>
                    <button class="tab-btn" onclick="switchTab('tab-history')">💬 Chat History Logs</button>
                    <button class="tab-btn" onclick="switchTab('tab-console')">💻 Live System Logs</button>
                </div>

                <!-- TAB 0: Chat Sandbox -->
                <div class="tab-content active" id="tab-sandbox">
                    <div class="chat-messages" id="chatBox">
                        <div class="msg bot">Welcome to JK Data Lab! I am your digital assistant, how can I help you?</div>
                    </div>
                    <div class="chat-input-area">
                        <input type="text" id="chatInput" placeholder="Type a message..." onkeydown="if(event.key==='Enter') sendTestChat()">
                        <button class="btn" onclick="sendTestChat()">Send</button>
                    </div>
                </div>

                <!-- TAB 1: System Config Form -->
                <div class="tab-content" id="tab-config">
                    <form id="configForm">
                        <div class="form-section-title">AI Engine & Model</div>
                        <div class="input-group">
                            <label for="ai_provider">Select LLM Provider</label>
                            <select id="ai_provider">
                                <option value="anthropic">Anthropic (Claude)</option>
                                <option value="openrouter">OpenRouter (Multi-model)</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="selected_model">AI Model Name</label>
                            <input type="text" id="selected_model" placeholder="e.g. claude-sonnet-5">
                        </div>
                        
                        <div class="form-section-title">API Keys & Tokens</div>
                        <div class="input-group">
                            <label for="anthropic_key">Anthropic API Key</label>
                            <input type="password" id="anthropic_key" placeholder="••••••••••••••••">
                        </div>
                        <div class="input-group">
                            <label for="openrouter_key">OpenRouter API Key</label>
                            <input type="password" id="openrouter_key" placeholder="••••••••••••••••">
                        </div>
                        <div class="input-group">
                            <label for="openai_key">OpenAI API Key (Required for vectorizing RAG)</label>
                            <input type="password" id="openai_key" placeholder="••••••••••••••••">
                        </div>
                        <div class="input-group">
                            <label for="meta_token">Meta Permanent Token</label>
                            <input type="password" id="meta_token" placeholder="••••••••••••••••">
                        </div>
                        <div class="input-group form-span-full">
                            <label for="meta_phone_id">Meta Phone Number ID</label>
                            <input type="text" id="meta_phone_id" placeholder="Meta Phone Number ID">
                        </div>

                        <div class="form-section-title">PostgreSQL Database Credentials</div>
                        <div class="input-group">
                            <label for="postgres_host">Host Name</label>
                            <input type="text" id="postgres_host" placeholder="localhost">
                        </div>
                        <div class="input-group">
                            <label for="postgres_port">Port</label>
                            <input type="text" id="postgres_port" placeholder="5432">
                        </div>
                        <div class="input-group">
                            <label for="postgres_user">Username</label>
                            <input type="text" id="postgres_user" placeholder="postgres">
                        </div>
                        <div class="input-group">
                            <label for="postgres_password">Password (Updated only if filled)</label>
                            <input type="password" id="postgres_password" placeholder="••••••••••••••••">
                        </div>
                        <div class="input-group form-span-full">
                            <label for="postgres_db">Database Name</label>
                            <input type="text" id="postgres_db" placeholder="postgres">
                        </div>

                        <div class="form-span-full" style="display:flex; justify-content:flex-end; margin-top: 1rem;">
                            <button type="button" class="btn" onclick="saveConfig()">Save Configurations</button>
                        </div>
                    </form>
                </div>

                <!-- TAB 2: Knowledge Base -->
                <div class="tab-content" id="tab-knowledge">
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem; line-height: 1.5;">
                        Upload documentation files (`.txt` format) to populate your business database. The system will vectorize the files and index them inside your PostgreSQL `pgvector` store.
                    </div>
                    <div id="indexedFilesBox" style="background: var(--bg-primary); border: 1px solid var(--outline); padding: 1rem; border-radius: 6px; font-size: 0.85rem; min-height: 120px; max-height: 200px; overflow-y: auto; margin-bottom: 1.2rem;">
                        <span style="color: var(--text-secondary);">Checking pgvector status...</span>
                    </div>
                    
                    <!-- File Upload Input Area -->
                    <div style="background: var(--bg-secondary); border: 1px dashed var(--outline); padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">
                        <label style="font-size: 0.75rem; color: var(--text-secondary); font-weight: 700; display:block; margin-bottom: 0.5rem; text-transform:uppercase;">Select Local Text File (.txt)</label>
                        <div style="display: flex; gap: 0.8rem; align-items: center;">
                            <input type="file" id="knowledgeUpload" accept=".txt" style="flex: 1; padding: 0.35rem 0.5rem; font-size: 0.8rem; background: var(--bg-primary); border: 1px solid var(--outline); border-radius: 4px; color: var(--text-secondary);">
                            <button class="btn" onclick="uploadFile()">Upload File</button>
                        </div>
                    </div>

                    <div style="display: flex; gap: 0.5rem; justify-content: flex-end;">
                        <button class="btn-secondary" onclick="loadKnowledgeStatus()">Reload Status</button>
                        <button class="btn" id="btnSyncIndex" onclick="reindexKnowledge()">Sync & Re-index pgvector</button>
                    </div>
                </div>

                <!-- TAB 3: Chat History Logs -->
                <div class="tab-content" id="tab-history">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem;">
                        <span style="font-size:0.85rem; color:var(--text-secondary)">Displays the latest persistent chatbot conversations logs:</span>
                        <button class="btn" style="padding: 0.4rem 1rem; font-size: 0.8rem;" onclick="loadLogs()">Refresh logs</button>
                    </div>
                    <div class="logs-container" id="logsBox">
                        <div style="color:var(--text-secondary); text-align:center; margin-top:3rem;">No logs found.</div>
                    </div>
                </div>

                <!-- TAB 4: Live System Logs -->
                <div class="tab-content" id="tab-console">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem;">
                        <span style="font-size:0.85rem; color:var(--text-secondary)">Real-time server process output (polls every 3 seconds):</span>
                        <button class="btn" style="padding: 0.4rem 1rem; font-size: 0.8rem;" onclick="loadSystemLogs()">Fetch logs</button>
                    </div>
                    <div id="sysLogsBox">
                        Loading live logs feed...
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentSelectedThread = null;

            // Load configuration and databases on load
            window.onload = function() {
                loadConfig();
                loadGlobalToggleState();
                loadInboxThreads();

                // Set up recurring update hooks
                setInterval(function() {
                    // Inbox now lives on the main homepage, so it always refreshes
                    loadInboxThreads();
                    if (currentSelectedThread) loadSelectedChat(currentSelectedThread);

                    const settingsModal = document.getElementById('settingsModal');
                    if (settingsModal.classList.contains('active')) {
                        const historyTab = document.getElementById('tab-history');
                        const consoleTab = document.getElementById('tab-console');

                        if (historyTab.classList.contains('active')) loadLogs();
                        if (consoleTab.classList.contains('active')) loadSystemLogs();
                    }
                }, 3000);
            };

            function loadGlobalToggleState() {
                fetch('/api/global-auto-reply')
                    .then(res => res.json())
                    .then(data => {
                        const toggle = document.getElementById('globalBotToggle');
                        toggle.checked = data.global_auto_reply;
                        updateToggleUI(data.global_auto_reply);
                    });
            }

            function handleGlobalToggle(elem) {
                const checked = elem.checked;
                fetch('/api/global-auto-reply', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ global_auto_reply: checked })
                })
                .then(res => res.json())
                .then(data => {
                    updateToggleUI(data.global_auto_reply);
                });
            }

            function updateToggleUI(isActive) {
                const text = document.getElementById('switchStatusText');
                if (isActive) {
                    text.innerText = "🤖 Bot Active";
                    text.style.color = "var(--tertiary)";
                } else {
                    text.innerText = "👤 Bot Paused";
                    text.style.color = "var(--text-secondary)";
                }
            }

            function openSettings() {
                document.getElementById('settingsModal').classList.add('active');
                switchTab('tab-sandbox');
            }

            function closeSettings(event) {
                document.getElementById('settingsModal').classList.remove('active');
            }

            function switchTab(tabId) {
                // Disable all tab contents and buttons
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                
                // Enable selected tab
                document.getElementById(tabId).classList.add('active');
                
                // Add active class to corresponding tab button
                const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.onclick.toString().includes(tabId));
                if (btn) btn.classList.add('active');
                
                // Instantly fetch data
                if (tabId === 'tab-history') loadLogs();
                if (tabId === 'tab-knowledge') loadKnowledgeStatus();
                if (tabId === 'tab-console') loadSystemLogs();
            }

            function loadConfig() {
                fetch('/api/config')
                    .then(res => res.json())
                    .then(data => {
                        document.getElementById('ai_provider').value = data.ai_provider;
                        document.getElementById('selected_model').value = data.selected_model;
                        document.getElementById('meta_phone_id').value = data.meta_phone_id;
                        
                        document.getElementById('postgres_host').value = data.postgres_host;
                        document.getElementById('postgres_port').value = data.postgres_port;
                        document.getElementById('postgres_user').value = data.postgres_user;
                        document.getElementById('postgres_db').value = data.postgres_db;
                        
                        if (data.has_anthropic_key) document.getElementById('anthropic_key').placeholder = "Key set (encrypted)";
                        if (data.has_openrouter_key) document.getElementById('openrouter_key').placeholder = "Key set (encrypted)";
                        if (data.has_openai_key) document.getElementById('openai_key').placeholder = "Key set (encrypted)";
                        if (data.has_meta_token) document.getElementById('meta_token').placeholder = "Token set (encrypted)";
                    });
            }

            function saveConfig() {
                const payload = {
                    ai_provider: document.getElementById('ai_provider').value,
                    selected_model: document.getElementById('selected_model').value,
                    meta_phone_id: document.getElementById('meta_phone_id').value,
                    anthropic_key: document.getElementById('anthropic_key').value,
                    openrouter_key: document.getElementById('openrouter_key').value,
                    openai_key: document.getElementById('openai_key').value,
                    meta_token: document.getElementById('meta_token').value,
                    
                    postgres_host: document.getElementById('postgres_host').value,
                    postgres_port: document.getElementById('postgres_port').value,
                    postgres_user: document.getElementById('postgres_user').value,
                    postgres_password: document.getElementById('postgres_password').value,
                    postgres_db: document.getElementById('postgres_db').value
                };

                fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                })
                .then(res => res.json())
                .then(data => {
                    alert(data.message);
                    loadConfig();
                })
                .catch(err => alert("Error updating configuration: " + err));
            }

            function loadLogs() {
                fetch('/api/logs')
                    .then(res => res.json())
                    .then(logs => {
                        const logsBox = document.getElementById('logsBox');
                        if (logs.length === 0) {
                            logsBox.innerHTML = '<div style="color:var(--text-secondary); text-align:center; margin-top:2rem;">No logs logged yet. Run tests to populate database.</div>';
                            return;
                        }
                        
                        logsBox.innerHTML = '';
                        logs.forEach(log => {
                            logsBox.innerHTML += `
                                <div class="log-item">
                                    <div class="log-header">
                                        <span><b>User Thread:</b> ${log.sender}</span>
                                        <span>${log.timestamp}</span>
                                    </div>
                                    <div class="log-body">
                                        <div class="log-text"><b>Query:</b> ${log.message}</div>
                                        <div class="log-text" style="color:var(--tertiary)"><b>AI Reply:</b> ${log.response}</div>
                                        <div style="margin-top:0.3rem; font-size:0.75rem;">
                                            <span class="badge ${log.provider}">${log.provider.toUpperCase()}</span>
                                            <span style="color:var(--text-secondary); margin-left: 0.4rem;">${log.model}</span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                    });
            }

            // WhatsApp Support Inbox dynamic rendering
            function loadInboxThreads() {
                fetch('/api/inbox/threads')
                    .then(res => res.json())
                    .then(threads => {
                        const listPane = document.getElementById('inboxThreadList');
                        if (threads.length === 0) {
                            listPane.innerHTML = '<span style="color:var(--text-secondary); text-align:center; margin-top:2rem; font-size:0.8rem;">No active WhatsApp conversations yet.</span>';
                            return;
                        }

                        listPane.innerHTML = '';
                        threads.forEach(t => {
                            const isSelect = currentSelectedThread === t.thread_id ? 'active-select' : '';
                            const replyBadge = t.auto_reply ? '<span class="badge-inbox bot-on">🤖 Bot Active</span>' : '<span class="badge-inbox bot-off">👤 Paused</span>';
                            const escBadge = t.status === 'escalated' ? '<span class="badge-inbox status-esc">🚨 Escalated</span>' : '';
                            
                            listPane.innerHTML += `
                                <div class="thread-card ${isSelect}" onclick="selectThread('${t.thread_id}')">
                                    <div class="thread-phone">${t.thread_id}</div>
                                    <div class="thread-status-line">
                                        ${replyBadge}
                                        ${escBadge}
                                    </div>
                                </div>
                            `;
                        });
                    });
            }

            function selectThread(threadId) {
                currentSelectedThread = threadId;
                loadInboxThreads();
                loadSelectedChat(threadId);
            }

            function loadSelectedChat(threadId) {
                fetch(`/api/inbox/thread/${threadId}`)
                    .then(res => res.json())
                    .then(data => {
                        const win = document.getElementById('inboxChatWindow');
                        
                        // Header with status flags
                        let autoReplyBtn = data.auto_reply ? 
                            `<button class="btn" style="background:var(--error); color:white; font-size:0.75rem; padding: 0.35rem 0.7rem;" onclick="toggleAutoReply('${threadId}', false, '${data.status}')">Pause Bot Auto-Reply</button>` :
                            `<button class="btn" style="background:var(--tertiary); color:var(--on-primary); font-size:0.75rem; padding: 0.35rem 0.7rem;" onclick="toggleAutoReply('${threadId}', true, '${data.status}')">Activate Bot Auto-Reply</button>`;
                        
                        let resolveBtn = data.status === 'escalated' ? 
                            `<button class="btn" style="background:var(--primary); color:var(--on-primary); font-size:0.75rem; padding: 0.35rem 0.7rem;" onclick="resolveEscalation('${threadId}')">Resolve Alert</button>` : '';

                        let msgListHtml = '';
                        data.messages.forEach(m => {
                            const isManual = m.provider === 'admin';
                            
                            // User Message Bubble
                            if (m.user_message && m.user_message !== '[Manual Admin message]') {
                                msgListHtml += `
                                    <div class="chat-bubble user">
                                        <div style="font-weight:600; font-size:0.7rem; color:var(--text-secondary); margin-bottom: 0.2rem;">User (${m.timestamp})</div>
                                        <div>${m.user_message}</div>
                                    </div>
                                `;
                            }
                            
                            // Bot response bubble
                            if (m.bot_response) {
                                const roleLabel = isManual ? 'Admin (Manual)' : 'Bot';
                                const roleClass = isManual ? 'admin' : 'bot';
                                const footerDetail = isManual ? '' : `<div style="font-size: 0.65rem; color:var(--text-secondary); margin-top:0.25rem;">${m.provider.toUpperCase()} | ${m.model}</div>`;
                                
                                msgListHtml += `
                                    <div class="chat-bubble ${roleClass}">
                                        <div style="font-weight:600; font-size:0.7rem; color:var(--primary); margin-bottom: 0.2rem;">${roleLabel} (${m.timestamp})</div>
                                        <div>${m.bot_response}</div>
                                        ${footerDetail}
                                    </div>
                                `;
                            }
                        });

                        win.innerHTML = `
                            <div class="chat-window-header">
                                <div>
                                    <div style="font-weight:700; color:var(--primary); font-size:0.95rem;">Chatting with: ${threadId}</div>
                                    <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.15rem;">
                                        Bot Control: ${data.auto_reply ? '🤖 Active Auto-Reply' : '👤 Manual Overridden'} | Status: ${data.status.toUpperCase()}
                                    </div>
                                </div>
                                <div style="display:flex; gap:0.4rem;">
                                    ${resolveBtn}
                                    ${autoReplyBtn}
                                </div>
                            </div>
                            
                            <div class="chat-window-messages" id="inboxMsgContainer">
                                ${msgListHtml || '<div style="color:var(--text-secondary); text-align:center; margin-top:4rem;">No messages logged.</div>'}
                            </div>

                            <div style="display:flex; gap:0.5rem; border-top: 1px solid var(--outline); padding-top: 0.8rem;">
                                <input type="text" id="manualReplyText" placeholder="Type a manual WhatsApp message to this customer..." style="flex:1; font-size:0.8rem;" onkeydown="if(event.key==='Enter') sendManualMessage('${threadId}')">
                                <button class="btn" onclick="sendManualMessage('${threadId}')">Send Manual Reply</button>
                            </div>
                        `;

                        // Auto-scroll messaging window to bottom
                        const msgContainer = document.getElementById('inboxMsgContainer');
                        msgContainer.scrollTop = msgContainer.scrollHeight;
                    });
            }

            function toggleAutoReply(threadId, autoReplyVal, statusVal) {
                fetch(`/api/inbox/thread/${threadId}/settings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ auto_reply: autoReplyVal, status: statusVal })
                })
                .then(res => res.json())
                .then(() => {
                    loadInboxThreads();
                    loadSelectedChat(threadId);
                });
            }

            function resolveEscalation(threadId) {
                fetch(`/api/inbox/thread/${threadId}/settings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ auto_reply: true, status: 'active' })
                })
                .then(res => res.json())
                .then(() => {
                    loadInboxThreads();
                    loadSelectedChat(threadId);
                });
            }

            function sendManualMessage(threadId) {
                const input = document.getElementById('manualReplyText');
                const text = input.value.trim();
                if (!text) return;

                input.value = 'Delivering...';
                input.disabled = true;

                fetch(`/api/inbox/thread/${threadId}/send`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success') {
                        loadSelectedChat(threadId);
                        loadInboxThreads();
                    } else {
                        alert("Failed to send: " + data.error);
                    }
                })
                .catch(err => alert("Error: " + err))
                .finally(() => {
                    input.value = '';
                    input.disabled = false;
                    input.focus();
                });
            }

            function loadKnowledgeStatus() {
                fetch('/api/knowledge/status')
                    .then(res => res.json())
                    .then(data => {
                        const box = document.getElementById('indexedFilesBox');
                        if (data.status === 'empty' || !data.files || data.files.length === 0) {
                            box.innerHTML = '<span style="color: var(--text-secondary);">Status: ⚠️ pgvector table is empty. Click sync to load data.</span>';
                        } else {
                            box.innerHTML = '<div><b>Indexed files in pgvector:</b></div>';
                            data.files.forEach(f => {
                                box.innerHTML += `<div style="padding-left: 0.5rem; margin-top: 0.25rem; color: var(--tertiary); font-size:0.8rem;">🗎 ${f}</div>`;
                            });
                        }
                    })
                    .catch(err => {
                        document.getElementById('indexedFilesBox').innerHTML = `<span style="color:var(--error);">Failed to load database status.</span>`;
                    });
            }

            function loadSystemLogs() {
                fetch('/api/system-logs')
                    .then(res => res.json())
                    .then(data => {
                        const box = document.getElementById('sysLogsBox');
                        const atBottom = box.scrollHeight - box.clientHeight <= box.scrollTop + 20;
                        
                        box.innerHTML = data.logs.join('\\n');
                        
                        if (atBottom) {
                            box.scrollTop = box.scrollHeight;
                        }
                    })
                    .catch(err => {
                        document.getElementById('sysLogsBox').innerText = "Failed to load log feed.";
                    });
            }

            function uploadFile() {
                const fileInput = document.getElementById('knowledgeUpload');
                if (fileInput.files.length === 0) {
                    alert("Please select a .txt file first.");
                    return;
                }

                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append("file", file);

                fetch('/api/knowledge/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(res => res.json())
                .then(data => {
                    alert(data.message);
                    fileInput.value = '';
                    loadKnowledgeStatus();
                })
                .catch(err => alert("Upload failed: " + err));
            }

            function reindexKnowledge() {
                const btn = document.getElementById('btnSyncIndex');
                const origText = btn.innerText;
                btn.innerText = "Indexing vectors...";
                btn.disabled = true;

                fetch('/api/knowledge/index', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    alert(data.message);
                    loadKnowledgeStatus();
                })
                .catch(err => alert("Error during pgvector indexing: " + err))
                .finally(() => {
                    btn.innerText = origText;
                    btn.disabled = false;
                });
            }

            function sendTestChat() {
                const input = document.getElementById('chatInput');
                const text = input.value.trim();
                if (!text) return;

                input.value = '';

                const chatBox = document.getElementById('chatBox');
                // Append user message
                chatBox.innerHTML += `<div class="msg user">${text}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                // Append loading placeholder
                const loadId = 'loading-' + Date.now();
                chatBox.innerHTML += `<div class="msg bot" id="${loadId}">Thinking...</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                })
                .then(res => res.json())
                .then(data => {
                    const placeholder = document.getElementById(loadId);
                    if (data.response) {
                        placeholder.innerText = data.response;
                    } else if (data.error) {
                        placeholder.innerText = "Error: " + data.error;
                        placeholder.style.color = "var(--error)";
                    }
                    chatBox.scrollTop = chatBox.scrollHeight;
                    loadLogs();
                })
                .catch(err => {
                    const placeholder = document.getElementById(loadId);
                    placeholder.innerText = "Error sending message: " + err;
                    placeholder.style.color = "var(--error)";
                });
            }
        </script>
    </body>
    </html>
    """

# ---------------------------------------------------------
# 3. WhatsApp Webhook POST/GET logic
# ---------------------------------------------------------

@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    """
    Handles Meta's GET verification handshake.
    """
    load_dotenv(override=True)
    verify_token = os.getenv("META_VERIFY_TOKEN")

    if mode == "subscribe" and token == verify_token:
        logger.info("Webhook verified successfully by Meta.")
        return PlainTextResponse(content=challenge, status_code=200)
    logger.warning(f"Webhook verification failed. Token mismatch: expected {verify_token}, got {token}")
    return Response(content="Verification failed", status_code=status.HTTP_403_FORBIDDEN)

@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Receives incoming customer messages, triggers LangGraph, and replies.
    """
    payload = await request.json()
    logger.info(f"Webhook POST received: {payload}")

    # Extract entry items
    entry = payload.get("entry", [])
    if not entry:
        return {"status": "empty_payload"}
        
    changes = entry[0].get("changes", [])
    if not changes:
        return {"status": "empty_changes"}
        
    value = changes[0].get("value", {})
    
    # Loop Prevention: Skip WhatsApp status updates (sent, delivered, read notifications)
    if "statuses" in value:
        return {"status": "ignored_status_update"}
        
    messages = value.get("messages", [])
    if not messages:
        return {"status": "no_messages"}
        
    message = messages[0]
    sender_phone = message.get("from")  # Customer's WhatsApp phone number
    message_type = message.get("type")

    # Guard against Meta redelivering the same message (e.g. after a slow response)
    message_id = message.get("id")
    if message_id and message_id in _recent_message_ids:
        logger.info(f"Ignoring duplicate webhook delivery for message {message_id}")
        return {"status": "duplicate_ignored"}
    if message_id:
        _recent_message_ids.append(message_id)

    # Process text messages
    if message_type == "text":
        user_message_text = message.get("text", {}).get("body", "")
        logger.info(f"Received message from {sender_phone}: {user_message_text}")
        
        # 1. Check global bot auto-reply status first
        load_dotenv()
        global_auto_reply = os.getenv("GLOBAL_BOT_AUTO_REPLY", "True") == "True"
        if not global_auto_reply:
            logger.info("Global Bot Auto-Reply is OFF. Logging message and skipping automatic response.")
            log_chat(sender_phone, user_message_text, "[Global Bot Paused - Manual support mode]", "human", "manual")
            return {"status": "global_bot_paused"}
            
        # 2. Fetch thread settings to check for auto-reply or escalation overrides
        settings = get_thread_settings(sender_phone)
        
        # 3. Check for escalation trigger keywords in the user's message
        escalation_keywords = ["admin", "human", "agent", "support", "help", "representative", "talk to person"]
        should_escalate = any(kw in user_message_text.lower() for kw in escalation_keywords)
        
        if should_escalate:
            # Notify the user that they are being escalated, pause the bot, and set status to escalated
            bot_response = "I have paused automatic replies and notified a human administrator. An agent will contact you shortly."
            set_thread_settings(sender_phone, auto_reply=False, status="escalated")
            send_whatsapp_message(sender_phone, bot_response)
            log_chat(sender_phone, user_message_text, bot_response, "bot", "escalation_override")
            logger.info(f"Thread {sender_phone} escalated to admin due to keyword trigger.")
            return {"status": "escalated"}
            
        # 4. If bot auto-reply is currently PAUSED, log the message but do not answer
        if not settings["auto_reply"]:
            logger.info(f"Thread {sender_phone} auto-reply is PAUSED. Logging message only.")
            log_chat(sender_phone, user_message_text, "[Manual Mode - Bot Paused]", "human", "manual")
            return {"status": "ignored_bot_paused"}
            
        # 5. Run LangGraph State Graph if auto-reply is active
        config = {"configurable": {"thread_id": sender_phone}}
        inputs = {"messages": [HumanMessage(content=user_message_text)]}
        output_state = chatbot_graph.invoke(inputs, config=config)
        
        # Get AI output message
        bot_response = output_state["messages"][-1].content
        logger.info(f"Bot response generated: {bot_response}")
        
        # Call Meta API to deliver message back to the customer
        send_whatsapp_message(sender_phone, bot_response)
        
        # Log to local transaction history SQLite database
        log_chat(
            sender_phone, 
            user_message_text, 
            bot_response, 
            os.getenv("AI_PROVIDER", "anthropic"), 
            os.getenv("SELECTED_MODEL", "")
        )
        
    return {"status": "processed"}


def send_whatsapp_message(to_number: str, text: str):
    """
    Sends a POST request to Meta's Graph API to deliver the text reply.
    """
    load_dotenv(override=True)
    phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
    meta_access_token = os.getenv("META_ACCESS_TOKEN")

    url = f"https://graph.facebook.com/v20.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {meta_access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {
            "body": text
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        if response.status_code != 200:
            logger.error(f"Failed to send message to {to_number}: {response_json}")
        else:
            logger.info(f"Message sent successfully to {to_number}")
    except Exception as e:
        logger.exception(f"Error while calling Meta Graph API: {e}")
