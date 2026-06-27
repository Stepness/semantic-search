import nltk
import os
from sentence_transformers import SentenceTransformer, util
import torch

def chunk_sliding_window(text, window=5, overlap=2):
    sentences = nltk.sent_tokenize(text)
    
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned.append(s)
    
    chunks = []
    i = 0
    while i < len(cleaned):
        window_sentences = cleaned[i:i+window]
        chunks.append({
            "text": " ".join(window_sentences),
            "start_sentence": i,
            "end_sentence": i + len(window_sentences) - 1
        })
        i += window - overlap
    
    return chunks

def chunk_by_paragraph(text, min_words=30):
    cleaned = []
    for s in text.split('\n\n'):
        s = s.strip()
        if s:
            cleaned.append(s)
    
    chunks = []
    buffer = ""
    
    for para in cleaned:
        word_count = len(para.split())
        if word_count < min_words:
            buffer += " " + para  # merge short paragraphs with next
        else:
            if buffer:
                para = buffer.strip() + " " + para
                buffer = ""
            chunks.append({
            "text": para,
        })
    
    if buffer:  # catch any remaining buffer
        chunks.append(buffer.strip())
    
    return chunks

def search_semantic(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    ranked_indices = torch.argsort(cosine_scores, descending=True).tolist()
    return ranked_indices[:top_k], cosine_scores.tolist()


with open("./books/rfc9110.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_sliding_window(text)
print(f"Total chunks: {len(chunks)}")

model_path = os.path.abspath('../all-MiniLM-L6-v2')
model = SentenceTransformer(model_path, local_files_only=True)
doc_embeddings = model.encode([c["text"] for c in chunks], convert_to_tensor=True)

while True:
    query = input("Type query: ")
    result, _ = search_semantic(query)
    for idx in result:
        print(f"- {chunks[idx]['text']}")