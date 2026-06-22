"""
Eval мульти-агента: 6 вопросов, три конфигурации:
  1) одиночный агент (agent_s5.run_agent)
  2) PWC без валидатора (orchestrator.run_pwc)
  3) PWC + валидатор (orchestrator.run_pwc с validate_plan)

Каждый вопрос прогоняется N раз в каждой конфигурации (default N=5).
Результат пишется в eval_pwc_results.json.

Запуск:
    python eval_pwc.py           # полный прогон, N=5
    python eval_pwc.py --single  # один прогон каждого кейса, быстрая проверка
    python eval_pwc.py -n 3      # N=3 прогона
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc

VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}

CASES = [
    # ===== Q1–Q3: исходные кейсы, на которых одиночный агент ломается =====
    {
        "id": "Q1",
        "query": (
            "Во сколько раз USD подорожал с 1 января 2022 по сегодня? "
            "В ответе используй аббревиатуру USD и слово «раз»."
        ),
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "require_tools": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["раз", "USD"],
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году? "
            "В ответе обязательно укажи знак %."
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "require_tools": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "В ответе укажи знак %."
        ),
        "comment": (
            "Класс ошибки D: одиночный галлюцинирует get_cumulative_inflation; "
            "PWC-планировщик тоже может — повод для валидатора."
        ),
        "require_tools": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
    },

    # ===== Q4: кейс, который чинит валидатор =====
    {
        "id": "Q4",
        "query": (
            "Какая средняя инфляция была за 2023 год? "
            "В ответе укажи знак %."
        ),
        "comment": (
            "PWC-планировщик с высокой вероятностью придумывает "
            "get_average_inflation или get_annual_inflation. "
            "Валидатор это ловит. Одиночный справляется сам через "
            "12 × get_inflation + calculate."
        ),
        "require_tools": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
    },

    # ===== Q5: естественная параллельность =====
    {
        "id": "Q5",
        "query": (
            "Какой курс был у USD, EUR и CNY на 1 января 2023? "
            "В ответе используй аббревиатуры USD, EUR, CNY."
        ),
        "comment": (
            "3 независимых get_fx_rate — для замера ускорения "
            "параллельного исполнения."
        ),
        "require_tools": {"get_fx_rate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
    },

    # ===== Q6: реальный макро-вопрос =====
    {
        "id": "Q6",
        "query": (
            "Сравни инфляцию за 2023 год и изменение курса USD за тот же период. "
            "Что росло быстрее? В ответе используй слово «инфляция», "
            "аббревиатуру USD и знак %."
        ),
        "comment": (
            "Реальный вопрос: сравнение инфляции и девальвации за 2023 год."
        ),
        "require_tools": {"get_fx_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["%", "инфляция", "USD"],
    },
]


def _check_single(case: dict, result: dict) -> dict:
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])

    required = case.get("require_tools", set())
    missing_tools = required - used

    ok = bool(ans) and not hallucinated and must and not missing_tools
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "missing_required_tools": sorted(missing_tools),
        "must_have_ok": must,
        "answer_preview": result.get("answer") or "",
    }


def _check_pwc(case: dict, result: dict) -> dict:
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES

    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])

    required = case.get("require_tools", set())
    missing_tools = required - used

    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
        and not missing_tools
    )
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "missing_required_tools": sorted(missing_tools),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": result.get("answer") or "",
        "trace": result.get("trace", []), 
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0, "times": []}
    pwc_no_val = {"runs": [], "pass": 0, "times": []}
    pwc_val = {"runs": [], "pass": 0, "times": []}

    for _ in range(n):
        # --- Одиночный агент ---
        t0 = time.time()
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "trace": [], "error": f"{type(e).__name__}: {e}"}
        t1 = time.time()
        single["times"].append(t1 - t0)
        
        check1 = _check_single(case, r1)
        single["runs"].append(check1)
        single["pass"] += int(check1["ok"])

        # --- PWC без валидатора ---
        t0 = time.time()
        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, validate=False)
        except Exception as e:
            r2 = {"answer": None, "trace": [], "plan": None,
                  "error": f"{type(e).__name__}: {e}"}
        t1 = time.time()
        pwc_no_val["times"].append(t1 - t0)
        
        check2 = _check_pwc(case, r2)
        pwc_no_val["runs"].append(check2)
        pwc_no_val["pass"] += int(check2["ok"])

        # --- PWC + валидатор ---
        t0 = time.time()
        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, validate=True)
        except Exception as e:
            r3 = {"answer": None, "trace": [], "plan": None,
                  "error": f"{type(e).__name__}: {e}"}
        t1 = time.time()
        pwc_val["times"].append(t1 - t0)
        
        check3 = _check_pwc(case, r3)
        pwc_val["runs"].append(check3)
        pwc_val["pass"] += int(check3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc_no_val": pwc_no_val,
        "pwc_val": pwc_val,
    }


def print_logs(results: list[dict], *, n: int) -> None:
    """Подробный лог по каждому кейсу и каждому прогону."""
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n\n{sep}")
    print("ПОДРОБНЫЕ ЛОГИ")
    print(sep)

    configs = [
        ("single",     "Одиночный агент"),
        ("pwc_no_val", "PWC без валидатора"),
        ("pwc_val",    "PWC + валидатор"),
    ]

    for r in results:
        print(f"\n{sep}")
        print(f"[{r['id']}] {r['query']}")
        print(f"    {r['comment']}")
        print(sep)

        for key, label in configs:
            block = r[key]
            print(f"\n  ── {label}  ({block['pass']}/{n} PASS) ──")
            for i, run in enumerate(block["runs"], 1):
                ok_mark = "✅" if run["ok"] else "❌"
                print(f"\n  Прогон {i} {ok_mark}")
                print(f"    used_tools : {run.get('used_tools') or '—'}")
                if run.get("plan_tools"):
                    print(f"    plan_tools : {run['plan_tools']}")
                if run.get("hallucinated"):
                    print(f"    ⚠ галлюцинации (worker)  : {run['hallucinated']}")
                if run.get("hallucinated_in_plan"):
                    print(f"    ⚠ галлюцинации (план)    : {run['hallucinated_in_plan']}")
                if run.get("hallucinated_in_workers"):
                    print(f"    ⚠ галлюцинации (workers) : {run['hallucinated_in_workers']}")
                if run.get("missing_required_tools"):
                    print(f"    ⚠ не хватает инструментов: {run['missing_required_tools']}")
                must_mark = "✅" if run.get("must_have_ok") else "❌"
                print(f"    must_have  : {must_mark}")
                if key != "single" and "iterations" in run:
                    print(f"    iterations : {run['iterations']}")
                preview = (run.get("answer_preview") or "").strip()
                if preview:
                    print(f"    answer     :")
                    for line in preview.splitlines():
                        print(f"      {line}")
                else:
                    print(f"    answer     : (пусто)")

        print(f"\n{thin}")


def main():
    ap = argparse.ArgumentParser(description="Eval PWC мульти-агента")
    ap.add_argument("--single", action="store_true",
                    help="Один прогон каждого кейса (быстрая проверка)")
    ap.add_argument("-n", type=int, default=5,
                    help="Число прогонов на кейс (default=5)")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval PWC: {len(CASES)} кейсов × {n} прогонов × 3 конфигурации\n")

    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        s = r["single"]
        p = r["pwc_no_val"]
        v = r["pwc_val"]
        print(f"   single: {s['pass']}/{n}  "
              f"pwc: {p['pass']}/{n}  "
              f"pwc+val: {v['pass']}/{n}")
        # Показываем галлюцинации в плане если есть
        for run in p["runs"] + v["runs"]:
            if run.get("hallucinated_in_plan"):
                print(f"   ⚠ план содержит выдуманные инструменты: "
                      f"{run['hallucinated_in_plan']}")
                break
        print()

    # Итоговая таблица
    print("=" * 70)
    print(f"{'ID':<4} {'single':>8} {'pwc':>8} {'pwc+val':>8}  вопрос")
    print("-" * 70)
    for r in results:
        print(f"{r['id']:<4} "
              f"{r['single']['pass']}/{n}{'':>4} "
              f"{r['pwc_no_val']['pass']}/{n}{'':>4} "
              f"{r['pwc_val']['pass']}/{n}{'':>4} "
              f" {r['query'][:50]}")
    print("=" * 70)

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nРезультаты: {out}")
    print_logs(results, n=n)

    # Время выполнения
    print("\n" + "=" * 70)
    print("СРЕДНЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ (сек):")
    print(f"{'ID':<4} {'single':>10} {'pwc':>10} {'pwc+val':>10}")
    print("-" * 70)
    for r in results:
        single_avg = sum(r["single"]["times"]) / len(r["single"]["times"]) if r["single"]["times"] else 0
        pwc_avg = sum(r["pwc_no_val"]["times"]) / len(r["pwc_no_val"]["times"]) if r["pwc_no_val"]["times"] else 0
        pwc_val_avg = sum(r["pwc_val"]["times"]) / len(r["pwc_val"]["times"]) if r["pwc_val"]["times"] else 0
        print(f"{r['id']:<4} {single_avg:>10.2f} {pwc_avg:>10.2f} {pwc_val_avg:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()