"""
schema.py — общие Pydantic-схемы пайплайна
===========================================
Заполняется постепенно, по мере прохождения раундов. На старте — пусто.

Карта моделей по раундам:
  Раунд 1   — Concern, Participant
  Раунд 2   — AspectSentiment, ParticipantSentiment
  Раунд 2.5 — DiscoveredAspects (для autodiscovery)
  Раунд 3   — ChunkSummary, DiscussionSummary
  Раунд 3.5 — GroupSummary (для иерархического Map-Reduce)
  Раунд 5   — ActionVerdict, JudgeReport
  Раунд 7   — MultiDocSummary
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════
class Concern(BaseModel):
    """Одна жалоба/проблема участника"""
    category: str = Field(description="Категория жалобы: ux/performance/support/price/security/other")
    text: str = Field(description="Краткое описание проблемы")
    quote: str = Field(description="ДОСЛОВНАЯ цитата из транскрипта, подтверждающая проблему")


class Participant(BaseModel):
    """Участник фокус-группы с его жалобами"""
    name: str = Field(description="Имя участника (Анна, Игорь, Дарья, Сергей)")
    age: int = Field(description="Возраст (число)")
    city: str = Field(description="Город проживания")
    profession: str = Field(description="Профессия/род занятий")
    concerns: list[Concern] = Field(description="Список жалоб/проблем участника")


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════

class AspectSentiment(BaseModel):
    """Оценка одного аспекта для участника"""
    aspect: Literal["price", "speed", "ux", "support", "security"] = Field(
        description="Аспект: price/ speed/ ux/ support/ security"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Тональность: positive/ neutral/ negative"
    )
    quote: str = Field(description="ДОСЛОВНАЯ цитата из транскрипта, подтверждающая оценку")


class ParticipantSentiment(BaseModel):
    """Все оценки одного участника"""
    name: str = Field(description="Имя участника")
    aspects: list[AspectSentiment] = Field(description="Список оценок по аспектам")


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════

class DiscoveredAspect(BaseModel):
    """Одна обнаруженная тема/аспект из транскрипта"""
    name: str = Field(description="Название темы (2-4 слова, на русском)")
    description: str = Field(description="Краткое описание, о чём эта тема")


class DiscoveredAspects(BaseModel):
    """Список тем, обнаруженных в транскрипте"""
    aspects: list[DiscoveredAspect] = Field(
        description="5-8 ключевых тем, которые обсуждали участники",
        min_items=5,
        max_items=8
    )


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce-резюме
# ══════════════════════════════════════════════════════════
class ChunkSummary(BaseModel):
    """Резюме одного фрагмента (одного говорящего)"""
    key_points: list[str] = Field(description="Ключевые тезисы участника")
    speaker: str = Field(description="Имя участника", default="unknown")
    sentiment: Literal["negative", "neutral", "positive"] = Field(
        description="Общая тональность высказываний"
    )


class DiscussionSummary(BaseModel):
    """Итоговое резюме всей дискуссии"""
    headline: str = Field(description="Заголовок (1 предложение)")
    key_findings: list[str] = Field(description="Ключевые выводы (5-7)")
    action_items: list[str] = Field(description="Рекомендации (3-5)")


# ══════════════════════════════════════════════════════════
# Раунд 3.5 — Иерархический Map-Reduce
# ══════════════════════════════════════════════════════════
class GroupSummary(BaseModel):
    """Резюме группы из 5-10 ChunkSummary (промежуточный уровень)"""
    group_id: int = Field(description="Номер группы")
    main_themes: list[str] = Field(description="Основные темы, поднятые в группе")
    key_insights: list[str] = Field(description="Ключевые инсайты (3-5)")
    sentiment_summary: Literal["mostly_negative", "mixed", "mostly_positive"] = Field(
        description="Общая тональность группы"
    )


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════
class ActionVerdict(BaseModel):
    """Вердикт по одной рекомендации"""
    action: str = Field(description="Текст рекомендации")
    support: Literal["supported", "weakly_supported", "not_supported"] = Field(
        description="Насколько рекомендация подтверждается данными"
    )
    evidence: list[str] = Field(description="Цитаты из participants.json, подтверждающие или опровергающие")
    comment: str = Field(description="Краткое объяснение вердикта")


class JudgeReport(BaseModel):
    """Полный отчёт судьи"""
    verdicts: list[ActionVerdict] = Field(description="Вердикты по каждой рекомендации")
    overall_score: float = Field(description="Общая оценка (0-1), среднее по вердиктам", ge=0, le=1)
    summary: str = Field(description="Краткое резюме проверки (1-2 предложения)")


# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc сводка
# ══════════════════════════════════════════════════════════

class MultiDocSummary(BaseModel):
    """Сводный анализ по нескольким банкам/документам"""
    common_themes: list[str] = Field(
        description="Темы, которые встречаются ВО ВСЕХ банках (общие проблемы)"
    )
    unique_per_bank: dict[str, list[str]] = Field(
        description="Уникальные проблемы/особенности каждого банка"
    )
    industry_recommendations: list[str] = Field(
        description="Рекомендации для всей индустрии (3-5)"
    )