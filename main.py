import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from config import DB_URL, MODEL_PATH, TOP_K

def get_db():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn, conn.cursor()

def search(cur, model, query, top_k=TOP_K):
    embedding = model.encode(query).tolist()
    cur.execute("""
        SELECT text, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding, embedding, top_k))
    return cur.fetchall()

if __name__ == "__main__":
    model = SentenceTransformer(os.path.abspath(MODEL_PATH), local_files_only=True)
    conn, cur = get_db()

    while True:
        query = input("Type query: ")
        for text, score in search(cur, model, query):
            print(f"[{score:.3f}] {text}\n")