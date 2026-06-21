"""
Pydantic-схемы для поиска фильмов по сюжету.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class MovieQuery(BaseModel):
    """Запрос пользователя о фильме (год и жанр опциональны)"""
    query: str = Field(..., min_length=3, description="Описание сюжета")
    year_hint: Optional[int] = Field(None, description="Предполагаемый год выпуска (опционально)")
    genre_hint: Optional[str] = Field(None, description="Предполагаемый жанр (опционально)")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError('Запрос должен содержать минимум 3 символа')
        return v.strip()
    
    @field_validator('year_hint')
    @classmethod
    def validate_year_hint(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            if v < 1888:
                raise ValueError(f'Год {v} слишком ранний для фильма (первый фильм - 1888)')
            if v > datetime.now().year:
                raise ValueError(f'Год {v} не может быть в будущем')
        return v


class MovieSearchResult(BaseModel):
    """Результат поиска фильма из RAG"""
    title: str = Field(..., min_length=1, description="Название фильма")
    year: int = Field(..., description="Год выпуска")
    plot: str = Field(..., min_length=10, description="Сюжет фильма")
    similarity_score: float = Field(..., ge=0, le=1, description="Сходство с запросом")
    chunk_id: str = Field(..., description="ID чанка в RAG")
    
    @field_validator('year')
    @classmethod
    def validate_movie_year(cls, v: int) -> int:
        if v < 1888:
            raise ValueError(f'Год {v} слишком ранний для фильма (первый фильм - 1888)')
        if v > datetime.now().year:
            raise ValueError(f'Год {v} не может быть в будущем')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if len(v.strip()) < 2:
            raise ValueError('Название фильма слишком короткое')
        return v.strip()


class WebSearchResult(BaseModel):
    """Результат поиска в интернете"""
    title: str = Field(..., description="Заголовок результата")
    url: str = Field(..., description="URL источника")
    snippet: str = Field(..., description="Фрагмент текста")
    source: str = Field(default="web", description="Источник")


class PlannerOutput(BaseModel):
    """Решение Планировщика"""
    reasoning: str = Field(..., description="Обоснование решения")
    rag_top_k: int = Field(..., ge=3, le=20, description="Сколько чанков взять из RAG")
    use_web_search: bool = Field(..., description="Нужен ли веб-поиск")
    web_search_query: Optional[str] = Field(None, description="Запрос для веб-поиска")
    has_year_hint: bool = Field(False, description="Есть ли подсказка по году")
    has_genre_hint: bool = Field(False, description="Есть ли подсказка по жанру")
    
    @field_validator('rag_top_k')
    @classmethod
    def validate_rag_k(cls, v: int) -> int:
        if v < 3:
            raise ValueError('Минимум 3 чанка для качественного поиска')
        if v > 20:
            raise ValueError('Максимум 20 чанков (ограничение контекста)')
        return v


class MovieAnswer(BaseModel):
    """Финальный ответ пользователю"""
    answer: str = Field(..., min_length=10, description="Ответ пользователю")
    title: str = Field(..., min_length=1, description="Название фильма")
    year: int = Field(..., description="Год выпуска")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность в ответе")
    rag_sources: list[str] = Field(default_factory=list, description="ID чанков из RAG")
    web_sources: list[str] = Field(default_factory=list, description="URL из веб-поиска")
    hallucination_score: int = Field(0, ge=0, description="Количество обнаруженных галлюцинаций")
    reasoning: str = Field(..., description="Краткое обоснование ответа")
    year_from_user: bool = Field(False, description="Год взят из подсказки пользователя")
    
    @field_validator('year')
    @classmethod
    def validate_movie_year(cls, v: int) -> int:
        if v < 1888:
            raise ValueError(f'Год {v} слишком ранний для фильма (первый фильм - 1888)')
        if v > datetime.now().year:
            raise ValueError(f'Год {v} не может быть в будущем')
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.3 and v >= 0:
            # Низкая уверенность - можно сказать "не знаю"
            pass
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if len(v.strip()) < 2:
            raise ValueError('Название фильма слишком короткое')
        return v.strip()


class CriticVerdict(BaseModel):
    """Вердикт Критика"""
    ok: bool = Field(..., description="Ответ принят или нет")
    confidence: float = Field(..., ge=0, le=1, description="Оценка уверенности")
    hallucination_count: int = Field(0, ge=0, description="Количество галлюцинаций")
    reason: str = Field(..., description="Объяснение решения")
    action: Literal["accept", "increase_rag", "rephrase_web"] = Field(
        ..., description="Что делать дальше"
    )
    rag_top_k_increase: int = Field(0, description="На сколько увеличить RAG k")
    rephrased_query: Optional[str] = Field(None, description="Перефразированный запрос для веб-поиска")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return v