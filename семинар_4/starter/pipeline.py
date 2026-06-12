"""
RAG: ChromaDB + OpenAI, поддержка разных стратегий чанкинга.

Команды:
    python pipeline.py ingest --strategy recursive
    python pipeline.py ingest --strategy fixed
    python pipeline.py ingest --strategy adaptive

    python pipeline.py ask "Кто сформулировал теорему Банаха?" --strategy recursive
    python pipeline.py ask "..." --strategy fixed
    python pipeline.py ask "..." --strategy adaptive

    python pipeline.py eval --strategy recursive
    python pipeline.py eval --strategy fixed
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_client import get_model, make_client
from rank_bm25 import BM25Okapi
from schema import RAGAnswer

# ============================================================
# Глобальные настройки
# ============================================================

client = make_client()
MODEL = get_model()
chroma = chromadb.PersistentClient(path="./chroma_db")

print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Эмбеддер готов за {time.time() - _t_embed:.1f}с", flush=True)

DATA_DIR = Path(__file__).parent / "data"
BM25_CACHE_TEMPLATE = Path(__file__).parent / "bm25_cache_{strategy}.json"


def get_collection_name(strategy: str) -> str:
    """Имя коллекции в ChromaDB в зависимости от стратегии."""
    return f"focus_groups_{strategy}"


def get_collection(strategy: str):
    """Получить коллекцию для заданной стратегии."""
    return chroma.get_or_create_collection(
        name=get_collection_name(strategy),
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def get_bm25_cache_path(strategy: str) -> Path:
    """Путь к BM25 кэшу для стратегии."""
    return Path(str(BM25_CACHE_TEMPLATE).format(strategy=strategy))


# ============================================================
# Стратегии чанкинга
# ============================================================

# Рекурсивный сплиттер
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)


def chunk_text_recursive(text: str) -> list[str]:
    """Стратегия B: рекурсивный сплиттер (по абзацам, предложениям)."""
    return [c.strip() for c in recursive_splitter.split_text(text) if c.strip()]


def chunk_text_fixed(text: str, chunk_size: int = 2000) -> list[str]:
    """Стратегия A: фиксированный размер, без перекрытия."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text_adaptive(text: str) -> list[str]:
    """
    Стратегия C: адаптивная — по пустым строкам (абзацам), группировка по 4 абзаца.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), 4):
        chunk = '\n\n'.join(paragraphs[i:i + 4])
        chunks.append(chunk)
    return chunks


# Словарь стратегий
CHUNK_STRATEGIES = {
    "recursive": chunk_text_recursive,
    "fixed": chunk_text_fixed,
    "adaptive": chunk_text_adaptive,
}


def chunk_text(text: str, strategy: str = "recursive") -> list[str]:
    """Выбрать стратегию чанкинга."""
    return CHUNK_STRATEGIES[strategy](text)


# ============================================================
# Токенизация
# ============================================================

def tokenize_ru(text: str):
    """Нормализация текста: приведение к нижнему регистру."""
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


# ============================================================
# Индексация
# ============================================================

def ingest(strategy: str = "recursive"):
    """Заполнение векторного хранилища и BM25 кэша для выбранной стратегии."""
    
    collection = get_collection(strategy)
    bm25_cache_path = get_bm25_cache_path(strategy)
    
    # Чистим старую коллекцию
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_chunks = []
    all_ids = []
    all_meta = []

    chunk_fn = CHUNK_STRATEGIES[strategy]

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_fn(text)

        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i, "strategy": strategy})

        print(f"  {f.stem}: {len(chunks)} чанков")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    bm25_data = {
        "strategy": strategy,
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    bm25_cache_path.write_text(json.dumps(bm25_data, ensure_ascii=False))

    total = collection.count()
    print(f"\nИндексировано ({strategy}): Dense — {total} чанков")
    print(f"BM25 — {len(all_ids)} чанков кэшировано в {bm25_cache_path.name}")


# ============================================================
# Поиск
# ============================================================

def _load_bm25(strategy: str):
    """Загрузить BM25 индекс для стратегии."""
    bm25_cache_path = get_bm25_cache_path(strategy)
    data = json.loads(bm25_cache_path.read_text())
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]


def retrieve(query: str, strategy: str = "recursive", k: int = 5) -> dict:
    """Dense-поиск в ChromaDB."""
    collection = get_collection(strategy)
    return collection.query(query_texts=[query], n_results=k)


def hybrid_retrieve(query: str, strategy: str = "recursive", k: int = 5, top: int = 15, c: int = 60) -> dict:
    """Hybrid-поиск контекста (dense + BM25 + RRF)."""
    
    # семантический поиск
    collection = get_collection(strategy)
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    # tf-idf поиск
    bm25, bm25_ids, bm25_texts = _load_bm25(strategy)
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)

    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    # reciprocal rank fusion
    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id[i] for i in top_ids]]}


# ============================================================
# Генерация ответа
# ============================================================

def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос по лекциям по математическому анализу. "
        "Опирайся ТОЛЬКО на контекст ниже. Если в контексте нет ответа — "
        "скажи об этом прямо.\n\n"
        "Правила:\n"
        "1. Опирайся ТОЛЬКО на контекст ниже. Не добавляй факты из общего знания.\n"
        "2. В `quotes` — 1-5 точных коротких цитат (НЕ пересказ).\n"
        "3. В `sources` — id блоков, откуда взяты цитаты (формат: 'транскрипция_матанализ_1__0').\n"
        "4. В `confidence` — честная оценка: 0.9+ ТОЛЬКО когда прямой ответ в контексте, "
        "0.5-0.8, если собран из нескольких кусков, < 0.5 — если контекст не отвечает на запрос.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str, strategy: str = "recursive"):
    """Задать вопрос с выбранной стратегией."""
    print(f"Поиск по базе (стратегия: {strategy})...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, strategy=strategy, k=15)
    found = hits["ids"][0]
    print(
        f"   нашёл {len(found)} чанков за {time.time() - t0:.1f}с: {', '.join(found)}",
        flush=True,
    )

    print("Генерация ответа...", flush=True)
    t1 = time.time()
    prompt = build_prompt(query, hits)
    resp: RAGAnswer = client.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(f"   ответ за {time.time() - t1:.1f}с", flush=True)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp)
    print("\n--- источники ---")
    for i in found:
        print(f"  {i}")


# ============================================================
# Eval
# ============================================================

def run_eval(strategy: str = "recursive", k: int = 5):
    """Запуск оценки для выбранной стратегии."""
    from eval import load_gold, hit_rate
    
    gold = load_gold()
    total = 0.0
    
    print(f"\n=== EVAL: стратегия {strategy}, hit-rate@{k} ===\n")
    
    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]
        
        hits = hybrid_retrieve(q, strategy=strategy, k=k)
        retrieved_ids = hits["ids"][0]
        
        score = hit_rate(retrieved_ids, gold_sources)
        total += score
        
        mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
        print(f"  [{item['id']:2d}] {item['type']:25s}  hit@{k} = {score:.2f}  {mark}  {q[:50]}...")
    
    mean = total / len(gold)
    print(f"\n  ИТОГО ({strategy}): hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")
    return mean


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline with multiple chunking strategies")
    parser.add_argument("command", choices=["ingest", "ask", "eval"],
                       help="Команда: ingest, ask, eval")
    parser.add_argument("query", nargs="?", help="Вопрос для ask")
    parser.add_argument("--strategy", "-s", choices=["recursive", "fixed", "adaptive"],
                       default="recursive", help="Стратегия чанкинга")
    parser.add_argument("--k", type=int, default=5, help="K для hit-rate@k")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest(strategy=args.strategy)
    elif args.command == "ask":
        if not args.query:
            print("Ошибка: нужен вопрос. Пример: python pipeline.py ask '...' --strategy recursive")
            sys.exit(1)
        ask(args.query, strategy=args.strategy)
    elif args.command == "eval":
        # Временный костыль — запускаем eval как подпроцесс
        import subprocess
        subprocess.run(["python", "eval.py", "--strategy", args.strategy, "--k", str(args.k)])


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()