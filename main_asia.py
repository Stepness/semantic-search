import requests
import zipfile
import io
import os

# Downloading and extracting the dataset from the compressed file
url = "https://github.com/gakudo-ai/open-datasets/raw/refs/heads/main/asia_documents.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("asia_data")

# Loading documents and getting their filenames
documents = []
doc_names = []
for file in os.listdir("asia_data"):
    if file.endswith(".txt"):
        with open(f"asia_data/{file}", "r", encoding="utf-8") as f:
            documents.append(f.read())
            doc_names.append(file)

print(f"Loaded {len(documents)} documents for the knowledge base.")

from rank_bm25 import BM25Okapi
import re

def tokenize_for_bm25(text):
    return (re.sub(r'[^\w\s]', '', text)).lower().split()

# BM25 requires that each text is tokenized as a (sub)list of words
tokenized_corpus = [tokenize_for_bm25(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)


def search_bm25(query, top_k=3):
    tokenized_query = query.lower().split()
    
    # Getting scores (lexical relevance to the query) for all documents
    scores = bm25.get_scores(tokenized_query)
    
    # Ranking documents by score
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked_indices[:top_k], scores



from sentence_transformers import SentenceTransformer, util
import torch
 
# Loading the pre-trained embedding model
model_path = os.path.abspath('../all-MiniLM-L6-v2')
model = SentenceTransformer(model_path, local_files_only=True)
 
# Pre-compute embeddings for our corpus (our "Vector DB")
# You do not need this step if you already have an external vector database:
# you may read and import your document vectors instead
doc_embeddings = model.encode(documents, convert_to_tensor=True)
 
def search_semantic(query, top_k=3):
    # Embedding the user's query into a vector
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculating cosine similarity between the query and all documents
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    # Ranking documents by similarity
    ranked_indices = torch.argsort(cosine_scores, descending=True).tolist()
    return ranked_indices[:top_k], cosine_scores.tolist()

def hybrid_search(query, top_k=3):
    # 1. Obtaining the two standalone search rankings
    bm25_ranks, _ = search_bm25(query, top_k=len(documents))
    semantic_ranks, _ = search_semantic(query, top_k=len(documents))
    
    # 2. Applying RRF formula: RRF_score = 1 / (k + rank)
    rrf_scores = {i: 0.0 for i in range(len(documents))}
    k_constant = 60  # The value of 60 is a standard academic convention
    
    # Adding RRF scores from BM25
    for rank, doc_idx in enumerate(bm25_ranks):
        rrf_scores[doc_idx] += 1.0 / (k_constant + rank + 1)
        
    # Adding RRF scores from semantic search
    for rank, doc_idx in enumerate(semantic_ranks):
        rrf_scores[doc_idx] += 1.0 / (k_constant + rank + 1)
    
    # 3. Sorting documents by their final fused RRF score
    final_ranked_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
    
    return final_ranked_indices[:top_k], rrf_scores

query = "countries with wrestling"

print(f"--- Query: '{query}' ---")

# Testing Semantic (good at understanding aspects like "nation-wise nuances" and conceptual titles)
print("\nTop Semantic Results:")
sem_indices, _ = search_semantic(query)
for idx in sem_indices:
    print(f"- {doc_names[idx]}")

# Testing BM25 (good at finding exact keyword-based matches like "rice", "field", "paddy")
print("\nTop BM25 Results:")
bm25_indices, _ = search_bm25(query)
for idx in bm25_indices:
    print(f"- {doc_names[idx]}")

# Testing Hybrid (balances both)
print("\nTop Hybrid (RRF) Results:")
hybrid_indices, _ = hybrid_search(query)
for idx in hybrid_indices:
    print(f"- {doc_names[idx]}")
    
# for name, tokens in zip(doc_names, tokenized_corpus):
#     if 'wrestling' in tokens:
#         print(f"{name}: FOUND wrestling")
#     else:
#         print(f"{name}: not found")
        
# tokenized_query = tokenize_for_bm25("which are the countries with wrestling")
# scores = bm25.get_scores(tokenized_query)
# for name, score in zip(doc_names, scores):
#     print(f"{name}: {score:.4f}")