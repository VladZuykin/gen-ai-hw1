"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик для поиска фильмов.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from llm_client import (
    get_model, make_client, 
    get_cost_tracker, reset_cost_tracker
)
from schemas import (
    MovieQuery, MovieAnswer, MovieSearchResult,
    PlannerOutput, CriticVerdict
)
from planner import planner
from critic import critic
from tools import search_rag, search_web, get_full_page_text
from utils import TraceLogger


class MovieOrchestrator:
    """Оркестратор поиска фильмов"""
    
    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_logger = TraceLogger(self.output_dir / "traces")
        self.max_attempts = 3
        self.client = make_client()
        
        # Пороги уверенности
        self.confidence_threshold = 0.7
        self.low_confidence_threshold = 0.4
        
    def run(self, query: str, year_hint: Optional[int] = None) -> dict[str, Any]:
        """Запустить поиск фильма по описанию сюжета"""
        
        reset_cost_tracker()
        
        start_time = time.time()
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        movie_query = MovieQuery(query=query, year_hint=year_hint)
        self.trace_logger.log_query(movie_query)
        
        print(f"\n{'='*60}")
        print(f"🔍 Поиск фильма: {query}")
        print(f"{'='*60}\n")
        
        # Шаг 1: Планировщик
        print("[Planner] Разработка стратегии...")
        plan = planner(query)
        print(f"  → {plan.reasoning}")
        print(f"  → RAG top_k: {plan.rag_top_k}")
        print(f"  → Веб-поиск: {'Да' if plan.use_web_search else 'Нет'}")
        self.trace_logger.log_plan(plan)
        
        # Шаг 2: Базовый поиск (только сниппеты)
        print(f"\n[Executor] Поиск в RAG (k={plan.rag_top_k})...")
        rag_results = search_rag(query, top_k=plan.rag_top_k)
        print(f"  → Найдено {len(rag_results)} результатов")
        
        web_results = []
        if plan.use_web_search:
            web_query = plan.web_search_query or query
            print(f"[Executor] Веб-поиск: {web_query}...")
            web_results = search_web(web_query, get_full_text=False)  # <-- ТОЛЬКО СНИППЕТЫ
            print(f"  → Найдено {len(web_results)} результатов")
        
        self.trace_logger.log_search_results(rag_results, web_results)
        
        # Шаг 3: Генерация ответа (с базовым контекстом)
        print(f"\n[Generator] Формирование ответа (базовый контекст)...")
        answer = self._generate_answer(query, rag_results, web_results, plan, year_hint)
        self.trace_logger.log_answer(answer)
        
        # Шаг 4: Проверяем уверенность и при необходимости добавляем контекст
        verdict = None
        full_text_loaded = False
        rag_increased = False
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n[Critic] Проверка (попытка {attempt}/{self.max_attempts})...")
            
            verdict = critic(query, answer, rag_results, web_results, attempt)
            self.trace_logger.log_verdict(verdict)
            
            print(f"  → Вердикт: {'✅ Принят' if verdict.ok else '❌ Отклонён'}")
            print(f"  → Уверенность: {verdict.confidence:.2f}")
            print(f"  → Галлюцинаций: {verdict.hallucination_count}")
            print(f"  → {verdict.reason}")
            
            # Если ответ хороший — принимаем
            if verdict.ok and verdict.confidence >= self.confidence_threshold:
                break
            
            # Если уверенность низкая и мы ещё не загружали полный текст
            if verdict.confidence < self.confidence_threshold and not full_text_loaded:
                print(f"\n[Generator] ⚠️ Уверенность {verdict.confidence:.2f} < {self.confidence_threshold}")
                print("   Загружаем полный текст со страниц для уточнения...")
                
                # Загружаем полный текст со страниц
                web_results_full = []
                for w in web_results:
                    if w.get('url'):
                        full_text = get_full_page_text(w['url'], max_chars=3000)
                        if full_text:
                            w_copy = dict(w)
                            w_copy['full_text'] = full_text
                            web_results_full.append(w_copy)
                        else:
                            web_results_full.append(w)
                    else:
                        web_results_full.append(w)
                
                web_results = web_results_full
                full_text_loaded = True
                
                # Перегенерируем ответ с полным текстом
                print("   Перегенерация ответа с полным текстом...")
                answer = self._generate_answer(query, rag_results, web_results, plan, year_hint)
                continue
            
            # Если уверенность всё ещё низкая и мы не увеличивали RAG
            if verdict.confidence < self.low_confidence_threshold and not rag_increased:
                new_k = min(plan.rag_top_k + 10, 30)
                print(f"\n[Executor] ⚠️ Уверенность {verdict.confidence:.2f} < {self.low_confidence_threshold}")
                print(f"   Увеличиваем RAG до {new_k}...")
                rag_results = search_rag(query, top_k=new_k)
                plan.rag_top_k = new_k
                rag_increased = True
                
                print("   Перегенерация ответа с расширенным RAG...")
                answer = self._generate_answer(query, rag_results, web_results, plan, year_hint)
                continue
            
            # Если уверенность низкая и все попытки исчерпаны
            if attempt == self.max_attempts:
                print(f"\n⚠️ Достигнут лимит попыток. Уверенность: {verdict.confidence:.2f}")
                break
            
            # Если критик предложил конкретное действие
            if verdict.action == "increase_rag" and not rag_increased:
                new_k = min(plan.rag_top_k + verdict.rag_top_k_increase, 30)
                print(f"\n[Executor] Увеличиваем RAG до {new_k}...")
                rag_results = search_rag(query, top_k=new_k)
                plan.rag_top_k = new_k
                rag_increased = True
                answer = self._generate_answer(query, rag_results, web_results, plan, year_hint)
                
            elif verdict.action == "rephrase_web":
                new_query = verdict.rephrased_query or query
                print(f"\n[Executor] Перефразируем веб-поиск: {new_query}...")
                web_results = search_web(new_query, get_full_text=full_text_loaded)
                answer = self._generate_answer(query, rag_results, web_results, plan, year_hint)
        
        # Шаг 5: Финальный ответ
        elapsed = time.time() - start_time
        tracker = get_cost_tracker()
        
        result = {
            "query": query,
            "year_hint": year_hint,
            "answer": answer.model_dump(),
            "rag_results": rag_results[:5],
            "web_results": web_results[:3],
            "verdict": verdict.model_dump() if verdict else None,
            "attempts": attempt if verdict else 0,
            "elapsed_seconds": round(elapsed, 2),
            "cost": tracker.to_dict(),
            "trace_id": trace_id,
            "full_text_loaded": full_text_loaded,
            "rag_increased": rag_increased,
        }
        
        self._save_result(result, trace_id)
        
        print(f"\n{'='*60}")
        print(f"✅ Найден фильм: {answer.title} ({answer.year})")
        print(f"   Уверенность: {answer.confidence:.2f}")
        print(f"   Попыток: {attempt if verdict else 0}")
        print(f"   Полный текст: {'✅ Да' if full_text_loaded else '❌ Нет'}")
        print(f"   Расширенный RAG: {'✅ Да' if rag_increased else '❌ Нет'}")
        print(f"   Время: {elapsed:.1f}с")
        tracker.print_summary()
        print(f"{'='*60}\n")
        
        return result
    
    def _generate_answer(
        self,
        query: str,
        rag_results: list,
        web_results: list,
        plan: PlannerOutput,
        year_hint: Optional[int] = None,
    ) -> MovieAnswer:
        """Сгенерировать ответ на основе результатов поиска"""
        
        # Формируем контекст из RAG
        rag_context = "\n\n".join([
            f"[{i+1}] {r['title']} ({r['year']}) - Сходство: {r['similarity_score']:.2f}\n"
            f"Сюжет: {r['plot'][:300]}..."
            for i, r in enumerate(rag_results[:10])
        ]) if rag_results else "(нет результатов RAG)"
        
        # Формируем контекст из веб-поиска
        web_parts = []
        for i, w in enumerate(web_results[:3]):
            part = f"[W{i+1}] {w['title']}\n{w['snippet'][:200]}..."
            # Если есть полный текст — добавляем его
            if w.get('full_text'):
                part += f"\nПолный текст: {w['full_text'][:500]}..."
            web_parts.append(part)
        
        web_context = "\n\n".join(web_parts) if web_parts else "(нет результатов веб-поиска)"
        
        year_hint_text = f" (подсказка пользователя: {year_hint})" if year_hint else ""
        
        prompt = f"""Ты — эксперт по поиску фильмов по описанию сюжета.

Запрос пользователя: {query}{year_hint_text}

Результаты поиска в RAG (похожие сюжеты):
{rag_context}

Результаты веб-поиска:
{web_context}

Найди фильм, который лучше всего соответствует описанию.

**ВАЖНЫЕ ПРАВИЛА:**
1. НЕ ПРИДУМЫВАЙ названия фильмов! Используй ТОЛЬКО названия из RAG или веб-поиска.
2. Если в веб-поиске есть точное название — используй ЕГО.
3. Если сомневаешься — скажи, что не знаешь.
4. Год обязательно должен быть корректным (1888-{datetime.now().year})
5. Ответ на РУССКОМ языке

Верни JSON с полями:
- title: название фильма (ТОЛЬКО из источников!)
- year: год выпуска
- answer: краткий ответ пользователю (2-3 предложения)
- confidence: уверенность 0-1
- rag_sources: список ID чанков из RAG
- web_sources: список URL из веб-поиска
- reasoning: почему выбран этот фильм
"""

        result = self.client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            response_model=MovieAnswer,
            temperature=0.2,
            max_retries=2,
            call_name="answer_generation",
        )
        
        return result
    
    def _save_result(self, result: dict, trace_id: str):
        """Сохранить результат в файл"""
        output_file = self.output_dir / f"{trace_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n📁 Результат сохранён: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python orchestrator.py \"описание сюжета\"")
        print("Пример: python orchestrator.py \"Парень узнаёт, что живёт в матрице\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    orchestrator = MovieOrchestrator()
    result = orchestrator.run(query)
    
    print("\n" + "=" * 60)
    print("🎬 ФИНАЛЬНЫЙ ОТВЕТ")
    print("=" * 60)
    print(f"Название: {result['answer']['title']} ({result['answer']['year']})")
    print(f"Ответ: {result['answer']['answer']}")
    print(f"Уверенность: {result['answer']['confidence']:.2f}")
    if result['answer']['hallucination_score']:
        print(f"⚠️ Галлюцинаций: {result['answer']['hallucination_score']}")
    print(f"Стоимость: ${result['cost']['total_cost_usd']:.6f}")
    print(f"Полный текст: {'✅ Да' if result.get('full_text_loaded') else '❌ Нет'}")
    print(f"Расширенный RAG: {'✅ Да' if result.get('rag_increased') else '❌ Нет'}")
    print("=" * 60)