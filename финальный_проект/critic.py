"""
Критик: проверяет качество ответа и решает, что делать дальше.
"""

from llm_client import get_model, make_client
from schemas import MovieAnswer, CriticVerdict


CRITIC_SYSTEM = """
Ты — критик системы поиска фильмов по сюжету.

Твоя задача — проверить, насколько хорош найденный ответ.

**Критерии оценки:**

1. **Уверенность (0-1)**:
   - 0.9-1.0: Сюжет идеально совпадает, все детали сходятся
   - 0.6-0.8: Сюжет похож, но есть расхождения
   - 0.3-0.5: Есть некоторые совпадения, но много неясного
   - 0.0-0.3: Сюжет не совпадает, вероятно, это другой фильм

2. **Галлюцинации**:
   - Проверь, есть ли НАЗВАНИЕ фильма в RAG или веб-результатах
   - Если название ВЫДУМАНО — это галлюцинация (hallucination_count +1)
   - Считай количество фактов в ответе, которые НЕ подтверждаются источниками

3. **Решение**:
   - accept: ответ хороший, уверенность ≥ 0.6, галлюцинаций мало
   - increase_rag: уверенность < 0.6 → увеличить количество чанков в RAG
   - rephrase_web: уверенность < 0.4 → перефразировать запрос для веб-поиска

**ПРАВИЛА:**
- ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ
- Если название фильма НЕ найдено в источниках — это ГАЛЛЮЦИНАЦИЯ
- Будь строгим, но справедливым
"""


def critic(
    query: str,
    answer: MovieAnswer,
    rag_results: list,
    web_results: list,
    attempt: int = 1
) -> CriticVerdict:
    """Критик: проверяет ответ и решает, что делать"""
    
    client = make_client()
    
    # Формируем контекст для критика
    rag_info = "\n".join([
        f"- {r['title']} ({r['year']}): совпадение {r['similarity_score']:.2f}"
        for r in rag_results[:5]
    ]) if rag_results else "(нет результатов RAG)"
    
    web_info = "\n".join([
        f"- {w['title']}: {w['snippet'][:100]}..."
        for w in web_results[:3]
    ]) if web_results else "(нет результатов веб-поиска)"
    
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"""
Запрос пользователя: {query}

Найденный ответ:
- Название: {answer.title}
- Год: {answer.year}
- Ответ: {answer.answer}
- Уверенность: {answer.confidence:.2f}
- Источники RAG: {answer.rag_sources}
- Источники веб: {answer.web_sources}

Результаты RAG (топ-5):
{rag_info}

Результаты веб-поиска:
{web_info}

Попытка: {attempt}

Оцени ответ и предложи действие.
"""}
    ]
    
    result = client.chat.completions.create(
        model=get_model(),
        messages=messages,
        response_model=CriticVerdict,
        temperature=0.0,
        max_retries=2,
        call_name="critic"
    )
    
    return result