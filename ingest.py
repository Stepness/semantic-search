import os
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from config import DB_URL, MODEL_PATH, CHUNK_MIN_WORDS, BATCH_SIZE

def chunk_by_paragraph(text, min_words=CHUNK_MIN_WORDS):
    chunks = []
    buffer = ""
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) < min_words:
            buffer += " " + para
        else:
            if buffer:
                para = buffer.strip() + " " + para
                buffer = ""
            chunks.append(para)
    if buffer:
        chunks.append(buffer.strip())
    return chunks

def get_db():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn, conn.cursor()

def ingest(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_by_paragraph(text)
    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer(os.path.abspath(MODEL_PATH), local_files_only=True)
    embeddings = model.encode(chunks, convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=True)

    conn, cur = get_db()
    cur.executemany(
        "INSERT INTO chunks (text, embedding) VALUES (%s, %s)",
        [(chunk, emb.tolist()) for chunk, emb in zip(chunks, embeddings)]
    )
    conn.commit()
    cur.close()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    abspath = os.path.abspath(sys.argv[1])
    ingest(abspath)