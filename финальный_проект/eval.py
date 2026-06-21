"""
Eval: оценка системы поиска фильмов на 15+ тестовых запросах.

Запуск:
    python eval.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from orchestrator import MovieOrchestrator
from llm_client import get_cost_tracker, reset_cost_tracker


# ============================================================================
# ТЕСТОВЫЕ ЗАПРОСЫ (15 штук)
# ============================================================================

TEST_QUERIES = [
    # === ТЕСТ 1: Матрица ===
    {
        "id": 1,
        "query": "Парень узнаёт, что живёт в матрице",
        "expected": {"title": "Матрица", "year": 1999}
    },
    
    # === ТЕСТ 2: Ведьма: Возрождение ===
    {
        "id": 2,
        "query": "Фильм, где в италии домик, который достался от деда, ужастик, ведьма, все умирают в конце",
        "expected": {"title": "Ведьма: Возрождение", "year": 2021}
    },
    
    # === ТЕСТ 3: 1+1 ===
    {
        "id": 3,
        "query": "Фильм, где один чернокожий, другой на инвалидной коляске",
        "expected": {"title": "1+1", "year": 2011}
    },
    
    # === ТЕСТ 4: Ведьма: Возрождение ===
    {
        "id": 4,
        "query": "Друзья пробуждают дух жуткой ведьмы из старинного итальянского особняка. Британский хоррор.",
        "expected": {"title": "Ведьма: Возрождение", "year": 2021}
    },
    
    # === ТЕСТ 5: Начало ===
    {
        "id": 5,
        "query": "Фильм про команду, которая внедряется в сны, чтобы украсть идею",
        "expected": {"title": "Начало", "year": 2010}
    },
    
    # === ТЕСТ 6: Люди в чёрном ===
    {
        "id": 6,
        "query": "Двое парней в строгих костюмах и тёмных очках ловят инопланетян, которые маскируются под людей",
        "expected": {"title": "Люди в чёрном", "year": 1997}
    },
    
    # === ТЕСТ 7: Побег из Шоушенка ===
    {
        "id": 7,
        "query": "Банкира сажают в тюрьму за убийство, которого он не совершал, он роет подкоп и выбирается",
        "expected": {"title": "Побег из Шоушенка", "year": 1994}
    },
    
    # === ТЕСТ 8: Форрест Гамп ===
    {
        "id": 8,
        "query": "Парень с низким IQ, но добрым сердцем, пробегает всю Америку и участвует в войне во Вьетнаме",
        "expected": {"title": "Форрест Гамп", "year": 1994}
    },
    
    # === ТЕСТ 9: Железный человек ===
    {
        "id": 9,
        "query": "Богатый оружейник, который много пьёт и говорит умные вещи, строит высокотехнологичный костюм, чтобы сбежать из плена в пещере",
        "expected": {"title": "Железный человек", "year": 2008}
    },
    
    # === ТЕСТ 10: Форсаж ===
    {
        "id": 10,
        "query": "Команда на быстрых машинах грабит грузовики, главный герой живёт ради семьи!!! и скорости, но полиция всегда на хвосте",
        "expected": {"title": "Форсаж", "year": 2001}
    },
    
    # === ТЕСТ 11: Титаник ===
    {
        "id": 11,
        "query": "Корабль, который считали непотопляемым, но он всё равно утонул, а влюблённые едва успевают попрощаться",
        "expected": {"title": "Титаник", "year": 1997}
    },
    
    # === ТЕСТ 12: Гарри Поттер ===
    {
        "id": 12,
        "query": "Мальчик-волшебник узнаёт, что он волшебник, и идёт в школу магии",
        "expected": {"title": "Гарри Поттер и философский камень", "year": 2001}
    },
    
    # === ТЕСТ 13: Властелин колец ===
    {
        "id": 13,
        "query": "Маленький хоббит должен уничтожить кольцо в вулкане, чтобы спасти мир",
        "expected": {"title": "Властелин колец: Братство кольца", "year": 2001}
    },
    
    # === ТЕСТ 14: Остров проклятых ===
    {
        "id": 14,
        "query": "Детектив приезжает на остров с психиатрической клиникой, расследует исчезновение пациентки",
        "expected": {"title": "Остров проклятых", "year": 2010}
    },
    
    # === ТЕСТ 15: Тёмный рыцарь ===
    {
        "id": 15,
        "query": "Богатый парень в костюме летучей мыши борется с безумным клоуном в городе",
        "expected": {"title": "Тёмный рыцарь", "year": 2008}
    },
]


# ============================================================================
# ФУНКЦИИ ДЛЯ ПРОВЕРКИ
# ============================================================================

def is_title_match(actual: str, expected: str) -> bool:
    """Проверить, совпадают ли названия (с учётом синонимов)"""
    actual_clean = actual.lower().strip()
    expected_clean = expected.lower().strip()
    
    # Точное совпадение
    if actual_clean == expected_clean:
        return True
    
    # Одно содержит другое
    if actual_clean in expected_clean or expected_clean in actual_clean:
        return True
    
    # Специальные случаи
    synonyms = {
        "1+1": ["1 + 1", "intouchables", "неприкасаемые"],
        "матрица": ["matrix"],
        "начало": ["inception"],
        "боец": ["fighter"],
    }
    
    for key, values in synonyms.items():
        if key in actual_clean and any(v in expected_clean for v in values):
            return True
        if key in expected_clean and any(v in actual_clean for v in values):
            return True
    
    return False


def is_year_match(actual: int, expected: int) -> bool:
    """Проверить, совпадают ли годы"""
    return actual == expected


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ EVAL
# ============================================================================

def run_eval(
    test_queries: List[Dict] = None,
    output_dir: Path = Path("output"),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Запустить eval на всех тестовых запросах.
    
    Args:
        test_queries: список тестов (если None, использует TEST_QUERIES)
        output_dir: папка для результатов
        verbose: показывать прогресс
    
    Returns:
        Результаты eval
    """
    if test_queries is None:
        test_queries = TEST_QUERIES
    
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = MovieOrchestrator(output_dir=output_dir)
    
    results = []
    passed = 0
    total_cost = 0.0
    total_time = 0.0
    total_tokens = 0
    
    print("=" * 70)
    print("🧪 ЗАПУСК EVAL")
    print("=" * 70)
    print(f"Всего тестов: {len(test_queries)}")
    print("=" * 70)
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected_title = test["expected"]["title"]
        expected_year = test["expected"]["year"]
        
        print(f"\n[{i}/{len(test_queries)}] {query[:60]}...")
        print("-" * 50)
        
        # Сбрасываем статистику перед каждым тестом
        reset_cost_tracker()
        
        start_time = time.time()
        
        try:
            # Запускаем оркестратор
            result = orchestrator.run(query)
            
            elapsed = time.time() - start_time
            
            # Извлекаем ответ
            actual_title = result["answer"]["title"]
            actual_year = result["answer"]["year"]
            confidence = result["answer"]["confidence"]
            hallucinations = result["answer"]["hallucination_score"]
            
            # Проверяем
            title_ok = is_title_match(actual_title, expected_title)
            year_ok = is_year_match(actual_year, expected_year)
            is_correct = title_ok and year_ok
            
            if is_correct:
                passed += 1
            
            # Собираем стоимость
            cost_data = result.get("cost", {})
            cost = cost_data.get("total_cost_usd", 0.0)
            tokens = cost_data.get("total_tokens", 0)
            total_cost += cost
            total_tokens += tokens
            total_time += elapsed
            
            # Результат теста
            test_result = {
                "id": test["id"],
                "query": query,
                "expected": {"title": expected_title, "year": expected_year},
                "actual": {"title": actual_title, "year": actual_year},
                "correct": is_correct,
                "title_match": title_ok,
                "year_match": year_ok,
                "confidence": confidence,
                "hallucinations": hallucinations,
                "attempts": result.get("attempts", 0),
                "elapsed_seconds": round(elapsed, 2),
                "cost_usd": round(cost, 6),
                "tokens": tokens,
                "full_text_loaded": result.get("full_text_loaded", False),
                "rag_increased": result.get("rag_increased", False),
                "rag_results": result.get("rag_results", [])[:3],
                "web_results": result.get("web_results", [])[:2],
                "verdict": result.get("verdict"),
            }
            results.append(test_result)
            
            # Вывод
            mark = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"\n  {mark}")
            print(f"    Ожидалось: {expected_title} ({expected_year})")
            print(f"    Получено:  {actual_title} ({actual_year})")
            print(f"    Уверенность: {confidence:.2f}")
            print(f"    Галлюцинаций: {hallucinations}")
            print(f"    Попыток: {test_result['attempts']}")
            print(f"    Время: {elapsed:.1f}с")
            print(f"    Стоимость: ${cost:.6f}")
            if test_result["full_text_loaded"]:
                print("    📄 Полный текст: Да")
            if test_result["rag_increased"]:
                print("    📚 RAG расширен: Да")
            
        except Exception as e:
            print(f"\n  ❌ ОШИБКА: {e}")
            results.append({
                "id": test["id"],
                "query": query,
                "expected": {"title": expected_title, "year": expected_year},
                "actual": {"title": None, "year": None},
                "correct": False,
                "error": str(e),
            })
    
    # ===== ИТОГИ =====
    total = len(test_queries)
    pass_rate = passed / total if total > 0 else 0
    
    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(pass_rate, 3),
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "total_time_seconds": round(total_time, 1),
        "avg_cost_per_test": round(total_cost / total, 6) if total > 0 else 0,
        "avg_time_per_test": round(total_time / total, 1) if total > 0 else 0,
        "tests": results,
    }
    
    # Сохраняем результаты
    output_file = output_dir / "eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Выводим итоги
    print("\n" + "=" * 70)
    print("📊 ИТОГИ EVAL")
    print("=" * 70)
    print(f"  Всего тестов:   {total}")
    print(f"  ✅ Пройдено:    {passed}")
    print(f"  ❌ Провалено:   {total - passed}")
    print(f"  📈 Pass rate:   {pass_rate:.1%}")
    print(f"  💰 Стоимость:   ${total_cost:.6f}")
    print(f"  🎯 Токенов:     {total_tokens:,}")
    print(f"  ⏱️  Время:       {total_time:.1f}с")
    print(f"  📊 Сред. стоимость: ${summary['avg_cost_per_test']:.6f} за тест")
    print("=" * 70)
    
    # Детальная таблица
    print("\n📋 ДЕТАЛЬНАЯ ТАБЛИЦА")
    print("-" * 70)
    print(f"{'ID':>3} | {'Результат':>6} | {'Фильм':<25} | {'Увер.':>5} | {'Галл.':>4} | {'Время':>5} | {'Стоим.':>8}")
    print("-" * 70)
    
    for r in results:
        mark = "✅ PASS" if r.get("correct") else "❌ FAIL"
        title = r.get("actual", {}).get("title", "N/A")[:22]
        conf = r.get("confidence", 0)
        hall = r.get("hallucinations", 0)
        elapsed = r.get("elapsed_seconds", 0)
        cost = r.get("cost_usd", 0)
        print(f"{r['id']:>3} | {mark:>6} | {title:<25} | {conf:>5.2f} | {hall:>4} | {elapsed:>5.1f} | ${cost:>7.6f}")
    
    print("-" * 70)
    print(f"\n📁 Результаты сохранены: {output_file}")
    
    return summary


# ============================================================================
# ЗАПУСК
# ============================================================================

def main():
    """Запуск eval"""
    # Можно запустить все тесты или только выборочные
    import argparse
    
    parser = argparse.ArgumentParser(description="Eval системы поиска фильмов")
    parser.add_argument(
        "--ids", 
        type=int, 
        nargs="+",
        help="ID тестов для запуска (например: --ids 1 2 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Папка для результатов"
    )
    args = parser.parse_args()
    
    # Фильтруем тесты по ID
    if args.ids:
        tests = [t for t in TEST_QUERIES if t["id"] in args.ids]
        if not tests:
            print(f"❌ Тесты с ID {args.ids} не найдены")
            print(f"   Доступные ID: {[t['id'] for t in TEST_QUERIES]}")
            sys.exit(1)
    else:
        tests = TEST_QUERIES
    
    run_eval(
        test_queries=tests,
        output_dir=Path(args.output),
        verbose=True
    )


if __name__ == "__main__":
    main()