"""
Сохраняет все чанки из bm25_cache в текстовые файлы для анализа.
"""

import json

# ============================================================
# Сохраняем fixed-чанки
# ============================================================
print("Загрузка bm25_cache_fixed.json...")
with open("bm25_cache_fixed.json", "r", encoding="utf-8") as f:
    data_fixed = json.load(f)

print(f"Сохранение fixed-чанков (всего {len(data_fixed['ids'])} шт.)...")
with open("chunks_fixed.txt", "w", encoding="utf-8") as out:
    for cid, text in zip(data_fixed["ids"], data_fixed["texts"]):
        out.write(f"{'='*80}\n")
        out.write(f"ID: {cid}\n")
        out.write(f"{'='*80}\n")
        out.write(text)
        out.write("\n\n")
print("Сохранено в chunks_fixed.txt")

# ============================================================
# Сохраняем recursive-чанки
# ============================================================
print("\nЗагрузка bm25_cache_recursive.json...")
with open("bm25_cache_recursive.json", "r", encoding="utf-8") as f:
    data_recursive = json.load(f)

print(f"Сохранение recursive-чанков (всего {len(data_recursive['ids'])} шт.)...")
with open("chunks_recursive.txt", "w", encoding="utf-8") as out:
    for cid, text in zip(data_recursive["ids"], data_recursive["texts"]):
        out.write(f"{'='*80}\n")
        out.write(f"ID: {cid}\n")
        out.write(f"{'='*80}\n")
        out.write(text)
        out.write("\n\n")
print("Сохранено в chunks_recursive.txt")

print("\nГотово!")