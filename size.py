import os
import glob
import sys
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import MODEL_PATH

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_by_paragraph(text, min_words=30):
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

filepaths = [os.path.abspath(p) for pattern in sys.argv[1:] for p in glob.glob(pattern)]
tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(MODEL_PATH), local_files_only=True)

token_counts = []
for filepath in filepaths:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_by_paragraph(text)
    token_counts.extend([count_tokens(c, tokenizer) for c in chunks])

print(f"Total chunks: {len(token_counts)}")
print(f"Min: {min(token_counts)}  Max: {max(token_counts)}  Avg: {sum(token_counts)//len(token_counts)}")

plt.figure(figsize=(14, 5))
n, bins, patches = plt.hist(token_counts, bins=20, edgecolor='black')
plt.xticks(bins, [str(int(b)) for b in bins], rotation=45, ha='right')
plt.xlabel("Tokens per chunk")
plt.ylabel("Number of chunks")
plt.title("Token distribution per paragraph")
plt.legend()
plt.tight_layout()
plt.savefig("token_distribution.png")
print("Saved to token_distribution.png")
counts = [(count_tokens(c, tokenizer), c) for c in chunks]
counts.sort(reverse=True)
print(f"Tokens: {counts[0][0]}\n\n{counts[0][1]}")