"""
Раунд 3.5 — Иерархический Map-Reduce
======================================
В раунде 3 REDUCE-промпт принимает ВСЕ мини-резюме сразу. Это работает
до тех пор, пока их сумма помещается в контекстное окно. Что если
участников 50 и каждое мини-резюме на 500 токенов? = 25k токенов
только в REDUCE. У DeepSeek окно 64k, у gpt-4o-mini 128k — пока влезает.
А если участников 500? Или каждое мини-резюме растёт с длиной
транскрипта?

Задача:
  Реализовать ТРЁХУРОВНЕВЫЙ Map-Reduce:

  • Уровень 1 (MAP):     фрагмент → ChunkSummary  (как в раунде 3)
  • Уровень 2 (GROUP):   группы по 5-10 ChunkSummary → GroupSummary
                          (новая модель — см. schema.py)
  • Уровень 3 (REDUCE):  все GroupSummary → DiscussionSummary

  На нашем коротком transcript эффект будет минимален (всего 4 фрагмента,
  иерархия избыточна). Но запустить и УВИДЕТЬ структуру — критично.
  В раунде 7 (многодокументном) на 5 транскриптах разница станет видимой.

Запуск:
    python 8_hierarchical_mr.py
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from llm_client import get_model, make_client
from prompts import GROUP_REDUCE_SYSTEM, REDUCE_SYSTEM
from schema import ChunkSummary, DiscussionSummary, GroupSummary

_mr = importlib.import_module("7_map_reduce")
summarize_chunk = _mr.summarize_chunk
_split_mod = importlib.import_module("6_split_chunking")
split_by_speaker = _split_mod.split_by_speaker

client = make_client()
MODEL = get_model()

GROUP_SIZE = 5


def reduce_group(group: list[ChunkSummary]) -> GroupSummary:
    """Уровень 2: 5-10 мини-резюме → одно групповое резюме.

    Подсказка: похоже на reduce_summaries из раунда 3, но
    response_model=GroupSummary (более лёгкая, чем DiscussionSummary).
    """
    # Склеиваем резюме группы
    group_text = f"## Группа из {len(group)} участников\n\n"
    for i, summary in enumerate(group, 1):
        group_text += f"### Участник {i} ({summary.speaker})\n"
        group_text += f"Тезисы: {', '.join(summary.key_points)}\n"
        group_text += f"Тональность: {summary.sentiment}\n\n"
    
    messages = [
        {"role": "system", "content": GROUP_REDUCE_SYSTEM},
        {"role": "user", "content": f"Агрегируй эти мини-резюме:\n\n{group_text}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=GroupSummary,
        max_retries=3,
        temperature=0.0
    )
    
    return result


def reduce_final(groups: list[GroupSummary]) -> DiscussionSummary:
    """Уровень 3: все групповые резюме → финальный DiscussionSummary."""
    # Склеиваем групповые резюме
    groups_text = "## Групповые резюме\n\n"
    for group in groups:
        groups_text += f"### Группа {group.group_id}\n"
        groups_text += f"Основные темы: {', '.join(group.main_themes)}\n"
        groups_text += f"Инсайты: {', '.join(group.key_insights)}\n"
        groups_text += f"Тональность: {group.sentiment_summary}\n\n"
    
    messages = [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content": f"Создай финальный свод на основе этих групповых резюме:\n\n{groups_text}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.0
    )
    
    return result


def hierarchical_summary(
    transcript: str, group_size: int = GROUP_SIZE
) -> DiscussionSummary:
    chunks = split_by_speaker(transcript)
    print(f"  [HMR] L1 MAP: {len(chunks)} фрагментов...")
    t0 = time.time()
    summaries = [summarize_chunk(c) for c in chunks]
    print(f"  [HMR] L1 готов ({time.time() - t0:.1f}с)")

    # Уровень 2: бьём на группы и сворачиваем
    groups_chunks = [
        summaries[i : i + group_size] for i in range(0, len(summaries), group_size)
    ]
    print(f"  [HMR] L2 GROUP: {len(groups_chunks)} групп...")
    t1 = time.time()
    groups = [reduce_group(g) for g in groups_chunks]
    print(f"  [HMR] L2 готов ({time.time() - t1:.1f}с)")

    # Уровень 3: финальный REDUCE
    print(f"  [HMR] L3 REDUCE: {len(groups)} групповых резюме...")
    t2 = time.time()
    final = reduce_final(groups)
    print(f"  [HMR] L3 готов ({time.time() - t2:.1f}с)")
    print(f"  [HMR] всего {time.time() - t0:.1f}с, {len(chunks)} → {len(groups)} → 1")
    return final


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    summary = hierarchical_summary(transcript)

    print("\n━━━ ИТОГ (иерархический) ━━━")
    print(summary.headline)
    for kf in summary.key_findings:
        print(f"  • {kf}")

    Path("summary_hierarchical.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print("\nСохранено: summary_hierarchical.json")
    print("Сравни с summary.json (раунд 3): иерархия сглаживает детали.")


if __name__ == "__main__":
    main()
