"""
Раунд 2 — Аспектный анализ + тепловая карта
=============================================
Теперь не просто «список жалоб», а структурированный взгляд: для каждого
участника — оценка по фиксированному набору аспектов (price/speed/ux/
support/security). На выходе — тепловая карта «участник × аспект».

Задача:
  1. В schema.py: AspectSentiment + ParticipantSentiment.
  2. В prompts.py: ASPECTS_SYSTEM (требование точной цитаты на русском,
     возврат только тех аспектов, что упомянуты).
  3. extract_aspects() — один вызов модели на весь транскрипт.
  4. build_heatmap() — тепловая карта (seaborn) участник × аспект.
  5. check_quotes() — на этом этапе ОБЯЗАТЕЛЬНО проверять цитаты:
     модель тут регулярно «сочиняет» (на DeepSeek типично 2-4 выдуманные
     цитаты).

Запуск:
    python 4_extract_aspects.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from llm_client import get_model, make_client
from prompts import ASPECTS_SYSTEM  # дополни prompts.py
from schema import ParticipantSentiment  # дополни schema.py

client = make_client()
MODEL = get_model()


def extract_aspects(transcript: str) -> list[ParticipantSentiment]:
    """Один запрос к модели → список оценок участников."""
    messages = [
        {"role": "system", "content": ASPECTS_SYSTEM},
        {"role": "user", "content": f"Проанализируй транскрипт и оцени каждого участника по аспектам:\n\n{transcript}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=list[ParticipantSentiment],
        max_retries=3,
        temperature=0.0
    )
    
    return result


def check_quotes(
    aspects: list[ParticipantSentiment],
    transcript: str,
) -> list[tuple[str, str]]:
    """Вернуть пары (имя, ghost-цитата) — те, что НЕ найдены в исходном тексте.

    Не пытайся искать дословно: модель может слегка переформулировать.
    Бери первые 30 символов цитаты в lowercase и ищи подстроку.
    """
    ghosts = []
    transcript_lower = transcript.lower()
    
    for p in aspects:
        for asp in p.aspects:
            quote = asp.quote
            if not quote:
                continue
            
            quote_clean = quote.strip()
            found = False
            
            # Стратегия 1: точное вхождение (без учёта регистра)
            if quote_clean.lower() in transcript_lower:
                found = True
            
            # Стратегия 2: первые 50 символов (убирая кавычки)
            if not found:
                quote_stripped = quote_clean.strip('"\'«»')
                if len(quote_stripped) > 20:
                    prefix = quote_stripped[:50].lower()
                    if prefix in transcript_lower:
                        found = True
            
            # Стратегия 3: ключевые слова (70% совпадения)
            if not found:
                words = set(w.lower() for w in quote_clean.split() if len(w) > 3)
                if words:
                    found_words = 0
                    for word in words:
                        if word in transcript_lower:
                            found_words += 1
                    
                    if found_words / len(words) > 0.7:
                        found = True
            
            if not found:
                ghosts.append((p.name, quote_clean[:80]))
    
    return ghosts


def build_heatmap(
    aspects: list[ParticipantSentiment],
    out_path: str = "heatmap.png",
) -> None:
    """Матрица participant × aspect, sentiment → {+1, 0, -1}, NaN если не упомянут."""
    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    
    # Список всех участников и аспектов
    participants = [p.name for p in aspects]
    all_aspects = ["price", "speed", "ux", "support", "security"]
    
    # Создаём матрицу
    matrix = np.full((len(participants), len(all_aspects)), np.nan)
    
    # Заполняем матрицу
    for i, p in enumerate(aspects):
        for asp in p.aspects:
            if asp.aspect in all_aspects:
                j = all_aspects.index(asp.aspect)
                matrix[i, j] = sentiment_map.get(asp.sentiment, 0)
    
    # Рисуем тепловую карту
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        matrix,
        xticklabels=all_aspects,
        yticklabels=participants,
        annot=True,
        fmt='.0f',
        cmap="RdYlGn",
        center=0,
        cbar_kws={'label': 'Sentiment (positive=1, neutral=0, negative=-1)'}
    )
    plt.title("Аспектный анализ: тональность по участникам")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Тепловая карта сохранена: {out_path}")


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")

    aspects = extract_aspects(transcript)
    print(
        f"Найдено: {len(aspects)} участников, всего "
        f"{sum(len(p.aspects) for p in aspects)} оценок."
    )

    ghosts = check_quotes(aspects, transcript)
    if ghosts:
        print(f"\n⚠ {len(ghosts)} цитат не найдено в транскрипте:")
        for name, q in ghosts[:5]:
            print(f"  {name}: «{q[:80]}»")

    build_heatmap(aspects)
    print("\nСохранено: heatmap.png")

    # Сохраним в JSON для следующих раундов.
    out = [p.model_dump() for p in aspects]
    Path("aspects.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Сохранено: aspects.json")


if __name__ == "__main__":
    main()
