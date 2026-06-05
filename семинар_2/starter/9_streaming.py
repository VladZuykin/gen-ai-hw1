from __future__ import annotations

import json
import sys
import time

from llm_client import get_model, make_raw_client
from prompts import SYSTEM_PROMPT, USER_PROMPT
from schema import Persona

client = make_raw_client()
MODEL = get_model()


def stream_and_validate() -> Persona:
    """Поток токенов на экран; в конце — валидация всего ответа."""
    print("⏵ ", end="", flush=True)
    t0 = time.time()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        stream=True,
        temperature=0.9,
    )

    chunks: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        sys.stdout.write(delta)
        sys.stdout.flush()

    dt = time.time() - t0
    print(f"\n  (поток закончен за {dt:.1f}с, символов: {sum(len(c) for c in chunks)})")

    raw = "".join(chunks).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return Persona.model_validate(json.loads(raw))


def main():
    print(f"Модель: {MODEL}")
    print("Запрос идёт в потоковом режиме — следи за «печатающей машинкой».\n")
    try:
        p = stream_and_validate()
        print(f"\n✓ Валидная персона: {p.name}, {p.age}, {p.address.city}")
    except Exception as e:
        print(f"\n✗ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()