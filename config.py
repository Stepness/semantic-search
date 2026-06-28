import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
MODEL_PATH = os.getenv("MODEL_PATH")
CHUNK_MIN_WORDS = 30
BATCH_SIZE = 8
TOP_K = 3