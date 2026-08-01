import os
import re
import logging
import httpx
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# PostgreSQL settings
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_NAME = os.getenv("POSTGRES_DB", "postgres")

# Directory containing knowledge documents
DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")

def get_db_connection():
    """
    Establishes connection to PostgreSQL database.
    """
    if not DB_PASSWORD:
        raise ValueError("PostgreSQL Password is not set. Please update your settings.")
        
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )

def init_pgvector():
    """
    Enables pgvector extension and creates the business_knowledge table.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        # Register vector type in psycopg2
        register_vector(conn)
        
        # 2. Create table for embeddings (using 1536 dimensions for OpenAI text-embedding-3-small)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS business_knowledge (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR(1536),
                file_name VARCHAR(255)
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("PostgreSQL pgvector database initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pgvector database: {e}")
        return False

def get_openai_embedding(text: str) -> list:
    """
    Generates a 1536-dimensional vector embedding for the input text using OpenAI API.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to OpenRouter key if present
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to generate vector embeddings.")
            
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    
    # Check if using OpenRouter fallback
    if api_key.startswith("sk-or-"):
        url = "https://openrouter.ai/api/v1/embeddings"
        payload["model"] = "openai/text-embedding-3-small"
        
    response = httpx.post(url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Splits document text into overlapping chunks for embedding.
    """
    # Clean text whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split(' ')
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap)
        
    return chunks

def index_knowledge_base():
    """
    Reads text files in the knowledge directory, generates embeddings, and saves them to PostgreSQL.
    """
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        # Create a sample text file if directory is empty
        sample_path = os.path.join(KNOWLEDGE_DIR, "knowledge.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(
                "JK Data Lab is a professional data engineering and AI automation consultancy company. "
                "Our offices are located in Mumbai, India. We are open Monday to Friday, 9:00 AM to 6:00 PM. "
                "We build WhatsApp Chatbots, automate social media workflows, and integrate custom databases for enterprises. "
                "Our contact email is contact@jkdatalab.com."
            )
            
    # Initialize the database table
    if not init_pgvector():
        return 0
        
    files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".txt")]
    if not files:
        logger.warning("No knowledge base .txt files found to index.")
        return 0
        
    indexed_count = 0
    try:
        conn = get_db_connection()
        register_vector(conn)
        cur = conn.cursor()
        
        # Clear existing index to avoid duplicate chunks on re-index
        cur.execute("TRUNCATE TABLE business_knowledge;")
        conn.commit()
        
        for file_name in files:
            file_path = os.path.join(KNOWLEDGE_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = split_text_into_chunks(content)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                # Generate vector embedding via API
                embedding = get_openai_embedding(chunk)
                
                # Insert chunk and vector into PostgreSQL pgvector table
                cur.execute(
                    "INSERT INTO business_knowledge (content, embedding, file_name) VALUES (%s, %s, %s);",
                    (chunk, embedding, file_name)
                )
                indexed_count += 1
                
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Indexed {indexed_count} chunks in pgvector successfully.")
        return indexed_count
    except Exception as e:
        logger.error(f"Error indexing knowledge base: {e}")
        raise e

def query_knowledge_base(query_text: str, limit: int = 3) -> str:
    """
    Searches the pgvector database for chunks most semantically similar to the query.
    Returns a compiled context string.
    """
    try:
        # Check if password is set
        if not DB_PASSWORD:
            logger.warning("PostgreSQL Password not set. Skipping RAG retrieval.")
            return ""
            
        # Get query embedding
        query_vector = get_openai_embedding(query_text)
        
        conn = get_db_connection()
        register_vector(conn)
        cur = conn.cursor()
        
        # Perform cosine distance query using pgvector '<=>' operator
        cur.execute("""
            SELECT content FROM business_knowledge
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (query_vector, limit))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            return ""
            
        context = "\n---\n".join([row[0] for row in rows])
        return context
    except Exception as e:
        logger.error(f"Error querying pgvector knowledge base: {e}")
        return ""

def get_indexed_files():
    """
    Returns list of indexed file names in pgvector.
    """
    try:
        if not DB_PASSWORD:
            return []
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT file_name FROM business_knowledge;")
        files = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return files
    except Exception:
        return []
