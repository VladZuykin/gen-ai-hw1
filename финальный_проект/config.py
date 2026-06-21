"""
Конфигурация проекта.
"""

from pathlib import Path

# Пути
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# RAG
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80

# Датасет с Kaggle
KAGGLE_DATASET = "maksimpotorochin/movie-plots-from-wikipedia-in-russian"

# Eval
TEST_QUERIES_FILE = DATA_DIR / "test_queries.json"