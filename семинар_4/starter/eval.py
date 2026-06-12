"""
Eval по gold-вопросам. Метрика: hit-rate@5 на уровне чанка.
"""

import argparse
import json
from pathlib import Path

from pipeline import hybrid_retrieve

# Базовый путь к gold-файлам
GOLD_BASE = Path(__file__).parent / "data"


def load_gold(strategy: str) -> list[dict]:
    """Загружает gold.json для конкретной стратегии"""
    if strategy == "fixed":
        gold_path = GOLD_BASE / "gold_fixed.json"
    elif strategy == "recursive":
        gold_path = GOLD_BASE / "gold.json"
    else:
        gold_path = GOLD_BASE / "gold.json"
    
    if not gold_path.exists():
        print(f"⚠ Файл {gold_path} не найден!")
        return []
    
    return json.loads(gold_path.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    if not gold_sources:
        return 1.0 if len(retrieved_ids) == 0 else 0.0
    found = [g for g in gold_sources if g in retrieved_ids]
    return len(found) / len(gold_sources)


def run(strategy: str = "recursive", k: int = 5, verbose: bool = True) -> dict:
    gold = load_gold(strategy)
    if not gold:
        return {"mean": 0.0, "results": []}
    
    total = 0.0
    results = []

    print(f"\n=== {strategy.upper()} ===\n")

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        hits = hybrid_retrieve(q, strategy=strategy, k=k)
        retrieved_ids = hits["ids"][0]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        if verbose:
            if not gold_sources:
                mark = "✓" if score == 1.0 else "✗"
                print(f"  [{item['id']:2d}] {item['type']:25s}  ловушка = {score:.2f}  {mark}  {q}")
            else:
                mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
                print(f"  [{item['id']:2d}] {item['type']:25s}  hit@{k} = {score:.2f}  {mark}  {q}")

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")
    return {"mean": mean, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", "-s", choices=["recursive", "fixed", "adaptive"], default="recursive")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run(strategy=args.strategy, k=args.k, verbose=not args.quiet)


if __name__ == "__main__":
    main()