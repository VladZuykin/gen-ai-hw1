from __future__ import annotations

import json

from llm_client import make_raw_client, get_model
from schema import Persona

client = make_raw_client()
MODEL = get_model()


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "register_persona",
        "description": "Зарегистрировать сгенерированную персону покупателя.",
        "parameters": Persona.model_json_schema(),
    },
}


def generate_via_tools() -> Persona:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ты генератор покупательских персон. Всегда используй инструмент register_persona для вывода результата."},
            {"role": "user", "content": "Создай одну персону покупателя."},
        ],
        tools=[TOOL_SCHEMA],
        temperature=0.9,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return Persona.model_validate(args)


def main():
    print(f"Модель: {MODEL}")
    print("━━━ Tool calling ━━━")
    for i in range(3):
        try:
            p = generate_via_tools()
            print(f"  [{i+1}/3] ✓ {p.name}, {p.age}, {p.city}")
        except Exception as e:
            print(f"  [{i+1}/3] ✗ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()