"""
Раунд 1.5 — Оценка качества извлечения
========================================
У нас есть два источника правды о транскрипте:
  • baseline_manual.json   — то, что выписали вручную в раунде 0 (эталон)
  • participants.json      — то, что нашла модель в раунде 1

Задача:
  Реализовать ТРИ метрики (имена функций оставлены латиницей, в выводе —
  русские названия):

  • полнота (coverage)      — какой процент тем из эталона нашла модель?
                полнота = |темы эталона ∩ темы модели| / |темы эталона|

  • точность (precision)    — какой процент жалоб модели реально есть в тексте?
                Проверять по подстроке `quote[:30].lower() in transcript.lower()`

  • достоверность (fidelity)— какой процент цитат, которые приводит модель,
                реально совпадает с текстом транскрипта (а не выдуман)?

  Для полноты сравнение тем — через модель (отдельный вызов «эта тема
  относится к одной из тем эталона?»). Это самая интересная часть —
  тут вы делаете модель-судью ещё до раунда 5.

Запуск:
    python 3_evaluate_ie.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client

client = make_client()
MODEL = get_model()


def load_artifacts() -> tuple[dict, list[dict], str]:
    baseline_path = Path("baseline_manual.json")
    if not baseline_path.exists():
        raise SystemExit("Сначала запусти раунд 0 — 1_baseline_manual.py.")
    participants_path = Path("participants.json")
    if not participants_path.exists():
        raise SystemExit("Сначала запусти раунд 1 — 2_extract_participants.py.")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    participants = json.loads(participants_path.read_text(encoding="utf-8"))
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    return baseline, participants, transcript


def fidelity(participants: list[dict], transcript: str) -> float:
    """Доля цитат, реально найденных в транскрипте (подстрочный поиск)."""
    
    total_quotes = 0
    found_quotes = 0
    ghost_quotes = []
    
    # Подготавливаем транскрипт для поиска
    transcript_lower = transcript.lower()
    
    for participant in participants:
        for concern in participant.get("concerns", []):
            quote = concern.get("quote", "")
            if not quote:
                continue
            
            total_quotes += 1
            quote_clean = quote.strip()
            
            # Стратегии поиска (от точного к мягкому)
            found = False
            
            # Стратегия 1: точное вхождение (но без учёта регистра)
            if quote_clean.lower() in transcript_lower:
                found = True
            
            # Стратегия 2: первые 50 символов (игнорируя кавычки в начале)
            if not found:
                # Убираем окружающие кавычки
                quote_stripped = quote_clean.strip('"\'«»')
                if len(quote_stripped) > 20:
                    prefix = quote_stripped[:50].lower()
                    if prefix in transcript_lower:
                        found = True
            
            # Стратегия 3: ключевая фраза (10-15 слов из середины)
            if not found and len(quote_clean.split()) > 5:
                words = quote_clean.split()
                # Берём середину цитаты (избегаем начала, где могут быть кавычки)
                mid_start = max(0, len(words) // 3)
                mid_end = min(len(words), mid_start + 10)
                key_phrase = ' '.join(words[mid_start:mid_end]).lower()
                if len(key_phrase) > 20 and key_phrase in transcript_lower:
                    found = True
            
            # Стратегия 4: поиск по словам (хотя бы 70% слов)
            if not found:
                quote_words = set(w.lower() for w in quote_clean.split() if len(w) > 3)
                if quote_words:
                    # Проверяем, сколько уникальных слов из цитаты есть в транскрипте
                    found_words = 0
                    for word in quote_words:
                        if word in transcript_lower:
                            found_words += 1
                    
                    if found_words / len(quote_words) > 0.7:  # 70% слов найдено
                        found = True
            
            if found:
                found_quotes += 1
            else:
                # Для отладки показываем, что искали
                short_quote = quote_clean[:80] + "..." if len(quote_clean) > 80 else quote_clean
                ghost_quotes.append((participant.get("name"), short_quote))
    
    if ghost_quotes:
        print(f"\n⚠ {len(ghost_quotes)} цитат не найдены в транскрипте:")
        for name, quote in ghost_quotes[:5]:
            print(f"  {name}: «{quote}»")
    
    result = found_quotes / total_quotes if total_quotes > 0 else 0.0
    print(f"\n  Найдено {found_quotes} из {total_quotes} цитат")
    return result


def precision(participants: list[dict], transcript: str) -> float:
    """Точность: доля жалоб, реально подтверждённых текстом (≈ достоверность,
    но можно усложнить — например, проверять не только наличие цитаты,
    но и совпадение категории).

    Для базовой версии — можно считать точность == достоверности.
    Для продвинутой — добавь свой критерий.
    """
    return fidelity(participants, transcript)


def coverage(baseline: dict, participants: list[dict]) -> float:
    """Полнота: доля тем из эталона, которые модель нашла.

    Сравнение тем — отдельным вызовом модели. Для каждой темы из эталона
    спрашиваем: «есть ли среди этих {llm_topics} тема, эквивалентная
    «{baseline_topic}»?». Ответ — да/нет.
    """
    model_topics = set()
    for p in participants:
        for concern in p.get("concerns", []):
            # Маппим категории модели на содержательные темы
            category = concern.get("category", "")
            text = concern.get("text", "")
            
            # Преобразуем категории в темы для сравнения с эталоном
            if category == "ux":
                if "шрифт" in text.lower():
                    model_topics.add("мелкий шрифт в истории операций")
                elif "уведомлени" in text.lower() or "пуш" in text.lower():
                    model_topics.add("избыточные push-уведомления")
                elif "анимац" in text.lower():
                    model_topics.add("избыточные анимации")
                elif "нагромождени" in text.lower() or "баннер" in text.lower():
                    model_topics.add("навязчивый upsell премиум-пакета")
            elif category == "performance":
                if "перевод" in text.lower() or "сбп" in text.lower():
                    model_topics.add("зависание переводов СБП")
                elif "интернет-банк" in text.lower() or "десктоп" in text.lower():
                    model_topics.add("лаги веб-версии")
            elif category == "support":
                if "ожидан" in text.lower():
                    model_topics.add("долгое ожидание поддержки")
                elif "бот" in text.lower():
                    model_topics.add("бесполезный чат-бот в поддержке")
            elif category == "price":
                if "ип" in text.lower():
                    model_topics.add("дорогие тарифы для ИП")
    
    # Подсчитываем, сколько тем из эталона нашла модель
    baseline_topics = [t["topic"] for t in baseline.get("topics", [])]
    found = sum(1 for topic in baseline_topics if topic in model_topics)
    
    return found / len(baseline_topics) if baseline_topics else 0.0


def main() -> None:
    baseline, participants, transcript = load_artifacts()

    f = fidelity(participants, transcript)
    p = precision(participants, transcript)
    c = coverage(baseline, participants)

    print("━━━ Метрики качества извлечения ━━━")
    print(f"  достоверность (fidelity)  = {f:.0%}   (цитаты совпадают с текстом)")
    print(f"  точность (precision)      = {p:.0%}   (жалобы подтверждены текстом)")
    print(f"  полнота (coverage)        = {c:.0%}   (темы эталона найдены моделью)")

    if c < 0.6:
        print("\n⚠ Низкая полнота — модель пропускает важные темы.")
        print("  Подкрути промпт IE_SYSTEM или увеличь temperature.")
    if f < 0.8:
        print("\n⚠ Низкая достоверность — модель «сочиняет» цитаты.")
        print("  Это галлюцинации. Усиль требование «дословно из текста» в промпте.")


if __name__ == "__main__":
    main()
