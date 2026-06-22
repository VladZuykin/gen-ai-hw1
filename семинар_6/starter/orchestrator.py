"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from validator import validate_plan
from worker import worker

def execute_level_parallel(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    max_workers: int = 4,
    verbose: bool = False,
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня параллельно."""
    results: dict[int, WorkerAnswer] = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sq in level:
            future = executor.submit(worker, sq, prev_answers)
            futures[future] = sq.id
        
        for future in as_completed(futures):
            sq_id = futures[future]
            try:
                ans = future.result()
                results[sq_id] = ans
                if verbose:
                    print(f"  [{sq_id}] ✓ {ans.answer[:60]}...")
            except Exception as e:
                results[sq_id] = WorkerAnswer(
                    subquestion_id=sq_id,
                    question_snippet="(ошибка)",
                    answer=f"(ошибка: {type(e).__name__}: {e})",
                    used_tools=[],
                    raw_trace=[],
                )
                if verbose:
                    print(f"  [{sq_id}] ❌ {e}")
    
    return results


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше."""

    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return None
        if node_id in path:
            raise ValueError(f"Цикл в depends_on : {path + [node_id]}")
        if node_id not in by_id:
            return None
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])
    return ordered

def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни для параллельного исполнения."""
    by_id = {s.id: s for s in subqs}
    
    # Проверка на циклы
    try:
        _topological_sort(subqs)
    except ValueError as e:
        raise ValueError(f"Цикл в depends_on: {e}")
    
    levels = []
    unassigned_ids = set(s.id for s in subqs)  # ← храним ID, а не объекты
    assigned_ids = set()
    
    while unassigned_ids:
        level = []
        for sq_id in list(unassigned_ids):
            sq = by_id[sq_id]
            if all(dep in assigned_ids for dep in sq.depends_on):
                level.append(sq)
                unassigned_ids.remove(sq_id)
        if not level:
            raise ValueError("Не удалось построить уровни (возможно, цикл)")
        levels.append(level)
        for sq in level:
            assigned_ids.add(sq.id)
    
    return levels


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    if not answers:
        return "(нет ответов)"

    from llm_client import make_client, get_model

    parts = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        parts.append(f"Подвопрос {sq_id}: {a.answer}")

    client = make_client()
    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": "Ты — макроэкономический аналитик. Собери ответы на подвопросы в 1-2 связные фразы для пользователя. Ответь числом с единицей измерения."
                },
                {
                    "role": "user",
                    "content": f"Исходный вопрос: {question}\n\nОтветы на подвопросы:\n" + "\n".join(parts)
                }
            ],
            temperature=0.0,
            max_retries=2,
        )
        return resp.choices[0].message.content or " · ".join([a.answer for a in answers.values()])
    except Exception:
        return " · ".join([a.answer for a in answers.values()])


def run_pwc(
    question: str, *, max_iter: int = 3, verbose: bool = True, validate: bool = False
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    if validate:  # ← только если validate=True
        errors = validate_plan(plan)
        if errors:
            if verbose:
                print(f"\n[validator] Ошибки: {errors}")
            plan = planner(question, feedback=f"Инструменты не существуют: {errors}")
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            level_answers = execute_level_parallel(
                level, answers, verbose=verbose
            )
            answers.update(level_answers)
            for sq_id, ans in level_answers.items():
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq_id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq_id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        # Обработка решений Критика
        if verdict.action == "replan":
            if verbose:
                print(f"  [orchestrator] 🔄 Перепланировка: {verdict.reason}")
            plan = planner(question, feedback=verdict.reason)
            continue

        elif verdict.action == "rework":
            if verbose:
                print(f"  [orchestrator] 🔄 Переделка подвопросов: {verdict.rework_ids}")
            for sq_id in verdict.rework_ids:
                sq = next((s for s in plan.subquestions if s.id == sq_id), None)
                if sq:
                    ans = worker(sq, prev_answers=answers)
                    answers[sq_id] = ans
            continue

        else:
            break

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
