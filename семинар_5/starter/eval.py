"""
Мини-оценка: 10 вопросов, проверяем:
1. Что агент завершает работу за разумное число шагов.
2. Что в трассе шагов есть ожидаемые инструменты.
3. Что в финальном ответе упомянуты ожидаемые ключевые числа (опционально).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    # ===== СТАРТОВЫЕ 4 ВОПРОСА =====
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Базовый тест — один инструмент, одно число.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика. Классический многостадийный кейс.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Вычисление с формулой: 72 / ставка = годы.",
    },

    # ===== 2 ВОПРОСА С compare_periods =====
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": ["раз"],
        "comment": "Требует compare_periods для USD. Агент должен использовать compare_periods вместо двух вызовов get_fx_rate.",
    },
    {
        "id": 6,
        "query": "Как изменилась инфляция между мартом 2024 и мартом 2025?",
        "expected_tools": ["compare_periods"],
        "must_have": ["%"],
        "comment": "Требует compare_periods для инфляции (cpi). Сравнение двух месяцев.",
    },

    # ===== 2 ТРУДНЫХ ВОПРОСА =====
    {
        "id": 7,
        "query": "Сравни курс USD в 2022-01 и 2026-04",
        "expected_tools": ["compare_periods"], 
        "must_have": ["доллар"],
        "comment": "ТРУДНЫЙ: неоднозначная дата (2022-01 может означать 1 января или весь январь). "
                    "Агент интерпретирует как 2022-01-01 и 2026-04-01. Это может не совпадать с ожиданиями пользователя.",
    },
    {
        "id": 8,
        "query": "Какая была инфляция в декабре 2023?",
        "expected_tools": ["get_inflation"],
        "must_have": ["%"],
        "comment": "ТРУДНЫЙ: декабрьские данные могут запаздывать или отсутствовать в CSV. "
                    "Если данных нет - агент должен вернуть ошибку, а не выдумывать число.",
    },

    # ===== 2 РЕАЛЬНЫХ МАКРО-ВОПРОСА =====
    {
        "id": 9,
        "query": "Что сейчас выше: ключевая ставка или инфляция?",
        "expected_tools": ["get_key_rate", "get_inflation"],
        "must_have": ["выше"],
        "comment": "РЕАЛЬНЫЙ: классический макро-вопрос - сравнение номинальной ставки и инфляции. "
                    "Важно для понимания реальной стоимости денег.",
    },
    {
        "id": 10,
        "query": "Сколько юаней можно купить за 100 долларов по курсу ЦБ сегодня?",
        "expected_tools": ["get_fx_rate", "calculate"],
        "must_have": ["юан"],
        "comment": "РЕАЛЬНЫЙ: кросс-курс валют через рубль. "
                    "Нужно получить курс USD/RUB и CNY/RUB, затем вычислить USD/CNY.",
    },
]


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    print(f"  comment: {case['comment']}")
    print(f"{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    # Проверяем ожидаемые инструменты (если указаны)
    tool_match = all(t in used_tools for t in case["expected_tools"])
    # Проверяем наличие ключевых слов
    text_match = all(s.lower() in answer.lower() for s in case["must_have"]) if case["must_have"] else True
    # Проверяем, что есть ответ
    has_answer = bool(answer)
    
    ok = has_answer and tool_match and text_match

    print(f"\n  tools used : {used_tools}")
    print(
        f"  expected    : {case['expected_tools']}  → {'OK' if tool_match else 'MISS'}"
    )
    print(f"  answer      : {answer[:200]}")
    if case["must_have"]:
        print(f"  must_have   : {case['must_have']}  → {'OK' if text_match else 'MISS'}")
    else:
        print(f"  must_have   : (не указаны)")
    print(f"  verdict     : {'PASS ✅' if ok else 'FAIL ❌'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
        "comment": case["comment"],
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Мини-оценка макро-агента")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Блок 9: общий кэш инструментов на все вопросы — видно повторные вызовы",
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="Блок 10: показать токены и стоимость по шагам",
    )
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    print(f"\n{'=' * 70}")
    print(f"ЗАПУСК EVAL: {len(CASES)} вопросов")
    print(f"{'=' * 70}")

    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in CASES]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}")
    print(f"ИТОГО: {passed}/{len(CASES)} пройдено")
    print(f"{'=' * 70}")
    
    for r in results:
        mark = "✅ PASS" if r["ok"] else "❌ FAIL"
        print(f"  {mark} Q{r['id']:2d} ({r['steps']} шагов) — {r['query'][:50]}...")
        if not r["ok"] and "comment" in r:
            print(f"         💡 {r['comment']}")

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(
            f"\n[кэш] на {len(CASES)} вопросах: {h} попаданий из {h + m} обращений "
            f"к инструментам — столько вызовов ЦБ/Росстата сэкономлено."
        )

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты сохранены в: {out}")


if __name__ == "__main__":
    main()