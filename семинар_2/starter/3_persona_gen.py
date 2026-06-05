"""
Семинар 2 — Лаборатория отладки
================================
Генератор синтетических персон для российского e-commerce.

Цель скрипта: вернуть 50 валидных персон и сохранить их в personas.json.

СТАТУС: скрипт УЖЕ падает. На каждом раунде семинара мы чиним по одной проблеме.
        Заметки «# TODO-раунд N» помечают места, куда придут изменения.
"""

import json
import time

from llm_client import get_model, make_client
from prompts import SYSTEM_PROMPT, USER_PROMPT
from schema import Persona

client = make_client()
MODEL = get_model()

N_PERSONAS = 50


def generate_one() -> dict:
    """Один запрос к LLM → одна персона."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        response_model=Persona,
        max_retries=3
    )

    return resp.model_dump()


def main():
    personas = []
    for i in range(N_PERSONAS):
        print(f"[{i + 1}/{N_PERSONAS}] запрос...")
        try:
            p = generate_one()
            personas.append(p)
            # Печатаем ТИП каждого поля — чтобы было видно, где плывёт.
            print(
                f"  → name={p.get('name')!r} "
                f"age={p.get('age')!r} (type={type(p.get('age')).__name__})"
            )
        except Exception as e:
            print(f"  ✗ упало: {type(e).__name__}: {e}")
        time.sleep(0.3)  # не долбим API

    # TODO-финал: после генерации полных 50 — `python 5_analysis.py` даёт
    #             расширенный отчёт (кросс-таблица, дубликаты, корреляции).

    print(f"\nСгенерировано: {len(personas)} из {N_PERSONAS}")
    with open("personas.json", "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print("Сохранено в personas.json")


if __name__ == "__main__":
    main()
