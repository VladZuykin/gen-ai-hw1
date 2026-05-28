"""
Параллельная генерация заявок на курсы с подсчетом бюджета
==========================================================
"""

from __future__ import annotations

import json
import time
import random
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import get_model, make_client
from prompts import get_system_prompt, USER_PROMPT
from schema import Application

client = make_client()
MODEL = get_model()

# Тарифы в долларах за 1M токенов (DeepSeek V4 Flash)
PRICE_INPUT_PER_1M = 0.14
PRICE_OUTPUT_PER_1M = 0.28

N_APPLICATIONS = 50
MAX_WORKERS = 10


def generate_one() -> tuple[Application, dict]:
    """Один запрос → (валидная заявка, словарь с usage)."""
    system_prompt = get_system_prompt()
    
    application, completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        response_model=Application,
        max_retries=3,
        temperature=1.2,
        seed=random.randint(1, 100000),
        with_completion=True,
    )
    
    usage = {
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
    }
    return application, usage


def estimate_cost(total_in: int, total_out: int) -> float:
    """Стоимость в долларах по двум счётчикам токенов."""
    return (total_in / 1_000_000) * PRICE_INPUT_PER_1M + (
        total_out / 1_000_000
    ) * PRICE_OUTPUT_PER_1M


def run_parallel(n: int, workers: int) -> tuple[list[Application], float, int, int]:
    """Параллельная генерация через ThreadPoolExecutor."""
    t0 = time.time()
    applications = []
    total_in, total_out = 0, 0
    done = 0
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_one) for _ in range(n)]
        
        for fut in as_completed(futures):
            try:
                app, usage = fut.result()
                applications.append(app)
                total_in += usage["input_tokens"]
                total_out += usage["output_tokens"]
                done += 1
                dt = time.time() - t0
                
                # Извлекаем данные для красивого вывода
                full_name = app.full_name
                city = app.address.city
                print(f"  [{done:02d}/{n}] {dt:.1f}с — {full_name}, {city}")
                
            except Exception as e:
                print(f"  ✗ ошибка: {type(e).__name__}: {e}")
                done += 1
    
    return applications, time.time() - t0, total_in, total_out


def save_results(applications: list[Application], output_json: str, output_csv: str):
    """Сохраняет результаты в JSON и CSV."""
    # Сохраняем JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([app.model_dump() for app in applications], f, ensure_ascii=False, indent=2)
    print(f"Сохранено в {output_json}")
    
    # Сохраняем CSV с распакованным address
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["full_name", "age", "city", "district", "speciality",
                         "desired_course", "years_of_experience", "graduation_year"])
        for app in applications:
            writer.writerow([
                app.full_name,
                app.age,
                app.address.city,
                app.address.district,
                app.speciality,
                app.desired_course,
                app.years_of_experience,
                app.graduation_year,
            ])
    print(f"Сохранено в {output_csv}")


def main():
    print(f"Модель: {MODEL}")
    print(f"Заявок: {N_APPLICATIONS}, воркеров: {MAX_WORKERS}\n")
    
    print("━━━ Параллельная генерация ━━━")
    applications, elapsed, total_in, total_out = run_parallel(N_APPLICATIONS, MAX_WORKERS)
    
    print("\n━━━ Сводка ━━━")
    print(f"Время:                    {elapsed:.1f}с")
    print(f"Валидных заявок:          {len(applications)}/{N_APPLICATIONS}")
    print(f"Входных токенов:          {total_in:>8d}")
    print(f"Выходных токенов:         {total_out:>8d}")
    
    cost = estimate_cost(total_in, total_out)
    print(f"Стоимость:                ${cost:.4f}")
    print(f"На 1 заявку:              ${cost / N_APPLICATIONS:.5f}")
    print(f"На 1000 заявок:           ${cost / N_APPLICATIONS * 1000:.2f}")
    
    if applications:
        save_results(applications, "applications.json", "applications.csv")
    
        # Статистика по городам
        cities = {}
        for app in applications:
            city = app.address.city
            cities[city] = cities.get(city, 0) + 1
        
        print("\n━━━ Распределение по городам ━━━")
        for city, count in sorted(cities.items(), key=lambda x: -x[1]):
            pct = count / len(applications) * 100
            status = "✅" if pct <= 40 else "⚠️ >40%"
            print(f"  {status} {city}: {count} ({pct:.1f}%)")
        
        # Статистика по специальностям
        print("\n━━━ Распределение по специальностям ━━━")
        specialities = {}
        for app in applications:
            spec = app.speciality
            specialities[spec] = specialities.get(spec, 0) + 1
        
        for spec, count in sorted(specialities.items(), key=lambda x: -x[1]):
            pct = count / len(applications) * 100
            status = "✅" if pct <= 35 else "⚠️ >35%"
            print(f"  {status} {spec}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()