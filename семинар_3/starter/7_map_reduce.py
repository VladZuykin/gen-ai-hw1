"""
Раунд 3b-d — Map-Reduce-резюме
=================================
Полный конвейер: разбиение → MAP в параллель → REDUCE в общий вывод.

Задача:
  1. В schema.py: ChunkSummary (key_points, speaker, sentiment) и
     DiscussionSummary (headline, key_findings, action_items).
  2. В prompts.py: CHUNK_SYSTEM и REDUCE_SYSTEM. Оба требуют русского.
  3. summarize_chunk()  — один вызов модели на один фрагмент.
  4. reduce_summaries() — один вызов модели на N мини-резюме.
  5. summarize_discussion() — собрать всё, MAP в ThreadPoolExecutor,
     потом REDUCE.

Запуск:
    python 7_map_reduce.py
"""

from __future__ import annotations

import importlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import get_model, make_client
from prompts import CHUNK_SYSTEM, REDUCE_SYSTEM
from schema import ChunkSummary, DiscussionSummary

_split_mod = importlib.import_module("6_split_chunking")
split_by_speaker = _split_mod.split_by_speaker

client = make_client()
MODEL = get_model()


def summarize_chunk(chunk: str) -> ChunkSummary:
    """MAP: один фрагмент → ChunkSummary."""
    speaker_match = re.match(r'^(Анна|Игорь|Дарья|Сергей):', chunk.strip())
    speaker = speaker_match.group(1) if speaker_match else "Неизвестный"
    
    messages = [
        {"role": "system", "content": CHUNK_SYSTEM},
        {"role": "user", "content": f"Проанализируй высказывания этого участника:\n\n{chunk}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.0
    )
    
    # Убеждаемся, что speaker установлен
    if result.speaker == "unknown":
        result.speaker = speaker
    
    return result


def reduce_summaries(summaries: list[ChunkSummary]) -> DiscussionSummary:
    """REDUCE: список мини-резюме → один общий свод.

    Склей summaries в строку (по одному блоку на фрагмент), отправь
    одним сообщением с response_model=DiscussionSummary.
    """
    # Склеиваем все мини-резюме в один текст
    summaries_text = []
    for i, s in enumerate(summaries, 1):
        summaries_text.append(f"## Участник {i} ({s.speaker})")
        summaries_text.append(f"Ключевые тезисы: {', '.join(s.key_points)}")
        summaries_text.append(f"Тональность: {s.sentiment}")
        summaries_text.append("")
    
    combined = "\n".join(summaries_text)
    
    messages = [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content": f"Создай единый свод на основе этих мини-резюме:\n\n{combined}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.0
    )
    
    return result


def summarize_discussion(transcript: str, workers: int = 6) -> DiscussionSummary:
    chunks = split_by_speaker(transcript)
    n = len(chunks)
    print(f"  [MR] MAP: {n} фрагментов, до {workers} параллельно...")
    t0 = time.time()

    summaries: list[ChunkSummary | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summarize_chunk, c): i for i, c in enumerate(chunks)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            summaries[i] = fut.result()
            done += 1
            print(f"  [MR] {done}/{n} готов ({time.time() - t0:.1f}с)")

    t_map = time.time() - t0
    print(f"  [MR] MAP {t_map:.1f}с → REDUCE...")
    result = reduce_summaries([s for s in summaries if s is not None])
    print(f"  [MR] всего {time.time() - t0:.1f}с")
    return result


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    summary = summarize_discussion(transcript)

    print("\n━━━ ИТОГ ━━━")
    print(summary.headline)
    print("\nКлючевые выводы:")
    for kf in summary.key_findings:
        print(f"  • {kf}")
    print("\nРекомендации:")
    for ai in summary.action_items:
        print(f"  → {ai}")

    Path("summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print("\nСохранено: summary.json")


if __name__ == "__main__":
    main()
