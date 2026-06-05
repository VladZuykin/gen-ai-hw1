"""
Раунд 2.5 — Автообнаружение аспектов
=====================================
В раунде 2 мы фиксировали аспекты в Literal — 5 заранее выбранных тем.
Это безопасно (модель не уйдёт в фантазии), но рискованно: если в
транскрипте обсуждали что-то ВНЕ нашего списка — мы это пропустим.

Задача:

  СТАДИЯ A. Обнаружение тем.
    Один вызов модели: «прочитай транскрипт, верни 5-8 ключевых тем,
    которые обсуждали». response_model=DiscoveredAspects (см. schema.py).
    Модель сама даёт список с описаниями.

  СТАДИЯ B. Классификация по найденным темам.
    Тот же extract_aspects, но aspect: str (не Literal), а в системный
    промпт вставляем список тем из стадии A. Получаем то же
    ParticipantSentiment, но с динамическим словарём.

Сравнить с раундом 2:
  • Сколько уникальных аспектов? (в раунде 2 ровно 5)
  • Появились ли темы, которых не было в Literal? Какие?
  • Пропали ли темы из Literal, которые на самом деле никто не обсуждал?

Запуск:
    python 5_aspect_discovery.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client
from prompts import ASPECTS_SYSTEM, DISCOVER_SYSTEM
from schema import (
    DiscoveredAspects,  # стадия A — что обнаружили
    ParticipantSentiment,  # стадия B — переиспользуем из раунда 2
)

client = make_client()
MODEL = get_model()


def discover_aspects(transcript: str) -> DiscoveredAspects:
    """Стадия A: что вообще обсуждали в этом транскрипте?"""
    lines = transcript.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if '═══' in line:
            start_idx = i + 1
            break
    
    # Берём только реплики участников (без модератора)
    participant_lines = []
    skip_moderator = False
    
    for line in lines[start_idx:]:
        if 'Модератор:' in line:
            skip_moderator = True
            continue
        elif line.strip() and not skip_moderator:
            participant_lines.append(line)
        elif 'Анна:' in line or 'Игорь:' in line or 'Дарья:' in line or 'Сергей:' in line:
            skip_moderator = False
            participant_lines.append(line)
    
    clean_transcript = '\n'.join(participant_lines)
    
    messages = [
        {"role": "system", "content": DISCOVER_SYSTEM},
        {"role": "user", "content": f"Выдели ключевые темы из этого транскрипта фокус-группы:\n\n{clean_transcript}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=DiscoveredAspects,
        max_retries=3,
        temperature=0.3  # Немного творчества для лучшего обнаружения
    )
    
    return result


def extract_with_discovered(
    transcript: str,
    discovered: DiscoveredAspects,
) -> list[ParticipantSentiment]:
    """Стадия B: те же оценки, но aspect — из обнаруженных тем, а не из Literal.

    Подсказка: дополни ASPECTS_SYSTEM динамическим куском —
        "Используй СТРОГО эти аспекты:\n- price (...)\n- speed (...)"
    и положи в system-промпт перед запросом.
    """
    # Формируем динамический промпт со списком обнаруженных тем
    aspects_list = "\n".join([
        f"- {a.name}: {a.description}" 
        for a in discovered.aspects
    ])
    
    dynamic_prompt = f"""Ты — анализатор тональности транскриптов фокус-групп.

ОБНАРУЖЕННЫЕ ТЕМЫ (используй ТОЛЬКО их):
{aspects_list}

ЗАДАЧА:
Для каждого участника определи, какие темы из списка выше он упоминал, и с какой тональностью.

ПРАВИЛА:
1. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
2. Используй ТОЛЬКО темы из предоставленного списка (не придумывай свои)
3. Указывай ТОЛЬКО темы, которые УПОМЯНУТЫ в речи участника
4. Каждая оценка ДОЛЖНА сопровождаться ДОСЛОВНОЙ цитатой из транскрипта
5. Тональность: positive/ neutral/ negative

ФОРМАТ ОТВЕТА:
Верни список участников, для каждого — список упомянутых тем с тональностью и цитатами.

ВАЖНО: НЕ добавляй темы, которых нет в списке выше!"""
    
    messages = [
        {"role": "system", "content": dynamic_prompt},
        {"role": "user", "content": f"Проанализируй транскрипт и оцени каждого участника по обнаруженным темам:\n\n{transcript}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=list[ParticipantSentiment],
        max_retries=3,
        temperature=0.0
    )
    
    return result


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")

    print("━━━ Стадия A: обнаружение тем ━━━")
    discovered = discover_aspects(transcript)
    print(f"Найдено тем: {len(discovered.aspects)}")
    for a in discovered.aspects:
        print(f"  • {a.name} — {a.description}")

    print("\n━━━ Стадия B: классификация по найденным темам ━━━")
    aspects = extract_with_discovered(transcript, discovered)
    print(f"Оценок: {sum(len(p.aspects) for p in aspects)}")

    # Сравнение с раундом 2 (если есть aspects.json):
    fixed_path = Path("aspects.json")
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
        fixed_aspects = {a["aspect"] for p in fixed for a in p["aspects"]}
        dyn_aspects = {a.aspect for p in aspects for a in p.aspects}
        new = dyn_aspects - fixed_aspects
        missing = fixed_aspects - dyn_aspects
        print(f"\nСравнение с раундом 2:")
        print(
            f"  было аспектов (Literal):     {len(fixed_aspects)} {sorted(fixed_aspects)}"
        )
        print(
            f"  стало (обнаруженные):        {len(dyn_aspects)} {sorted(dyn_aspects)}"
        )
        if new:
            print(f"  ⊕ новые темы: {sorted(new)}")
        if missing:
            print(
                f"  ⊖ пропали (Literal-темы, которых на самом деле не обсуждали): {sorted(missing)}"
            )

    Path("aspects_discovered.json").write_text(
        json.dumps([p.model_dump() for p in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nСохранено: aspects_discovered.json")


if __name__ == "__main__":
    main()
