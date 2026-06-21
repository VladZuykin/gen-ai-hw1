"""
Исполнитель: генерирует ответ на основе результатов поиска.
"""

from datetime import datetime
from typing import List, Dict, Any

from llm_client import make_client, get_model
from schemas import MovieAnswer, PlannerOutput


def generate_answer(
    query: str,
    rag_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    plan: PlannerOutput,
) -> MovieAnswer:
    """
    Сгенерировать ответ на основе результатов RAG и веб-поиска.
    
    Args:
        query: исходный запрос пользователя
        rag_results: результаты из RAG
        web_results: результаты из веб-поиска
        plan: план от Планировщика
    
    Returns:
        MovieAnswer: структурированный ответ
    """
    client = make_client()
    
    # Формируем контекст из RAG
    rag_context = "\n\n".join([
        f"[{i+1}] {r['title']} ({r['year']}) - Сходство: {r['similarity_score']:.2f}\n"
        f"Сюжет: {r['plot'][:300]}..."
        for i, r in enumerate(rag_results[:10])
    ]) if rag_results else "(нет результатов RAG)"
    
    # Формируем контекст из веб-поиска
    web_context = "\n\n".join([
        f"[W{i+1}] {w['title']}\n{w['snippet'][:200]}..."
        for i, w in enumerate(web_results[:3])
    ]) if web_results else "(нет результатов веб-поиска)"
    
    year_hint_text = f" (подсказка пользователя: {plan.year_hint})" if plan.year_hint else ""
    
    prompt = f"""Ты — эксперт по поиску фильмов по описанию сюжета.

Запрос пользователя: {query}{year_hint_text}

Результаты поиска в RAG (похожие сюжеты):
{rag_context}

Результаты веб-поиска:
{web_context}

Найди фильм, который лучше всего соответствует описанию.

Правила:
1. Если есть явное совпадение — укажи его
2. Если совпадений несколько — выбери самое похожее
3. Если нет уверенности — скажи об этом честно
4. Год обязательно должен быть корректным (1888-{datetime.now().year})
5. Ответ на РУССКОМ языке

Верни JSON с полями:
- title: название фильма
- year: год выпуска
- answer: краткий ответ пользователю (2-3 предложения)
- confidence: уверенность 0-1
- rag_sources: список ID чанков из RAG
- web_sources: список URL из веб-поиска
- reasoning: почему выбран этот фильм
"""

    result = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": prompt}],
        response_model=MovieAnswer,
        temperature=0.2,
        max_retries=2,
        call_name="answer_generation", 
    )
    
    return result