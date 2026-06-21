"""
Фабрика OpenAI-совместимого клиента + лёгкий JSON-инструктор + встроенный трекинг стоимости.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Type, TypeVar, get_args, get_origin
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter

# .env загрузим, если есть python-dotenv
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

T = TypeVar("T")

# ============================================================================
# ГЛОБАЛЬНЫЙ ТРЕКЕР СТОИМОСТИ
# ============================================================================

@dataclass
class CostTracker:
    """Глобальный трекер стоимости всех вызовов LLM"""
    
    # Цены моделей (за 1M токенов, USD)
    MODEL_PRICES = {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19},
        "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "default": {"input": 0.14, "output": 0.28},
    }
    
    # Статистика
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    calls: list[dict] = field(default_factory=list)
    _last_call: dict | None = None
    
    def add_call(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "deepseek-chat",
        extra: dict | None = None
    ) -> float:
        """
        Добавить вызов LLM.
        
        Args:
            name: имя шага (planner, critic, answer_generation)
            prompt_tokens: входные токены
            completion_tokens: выходные токены
            model: имя модели
            extra: дополнительные данные (например, температура)
        
        Returns:
            Стоимость вызова в USD
        """
        prices = self.MODEL_PRICES.get(model, self.MODEL_PRICES["default"])
        
        input_cost = prompt_tokens / 1_000_000 * prices["input"]
        output_cost = completion_tokens / 1_000_000 * prices["output"]
        cost = input_cost + output_cost
        
        self.total_cost += cost
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        
        call_record = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost, 8),
            "model": model,
            "extra": extra or {},
        }
        self.calls.append(call_record)
        self._last_call = call_record
        
        return cost
    
    def get_last_call(self) -> dict | None:
        """Вернуть последний вызов"""
        return self._last_call
    
    def to_dict(self) -> dict:
        """Вернуть полную статистику"""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "calls": self.calls,
        }
    
    def print_summary(self):
        """Вывести краткую сводку"""
        if not self.calls:
            print("  (нет вызовов)")
            return
        
        print(f"\n  📊 СТАТИСТИКА СТОИМОСТИ")
        print(f"     Всего токенов: {self.total_tokens:,}")
        print(f"     Входных:       {self.total_prompt_tokens:,}")
        print(f"     Выходных:      {self.total_completion_tokens:,}")
        print(f"     Стоимость:     ${self.total_cost:.6f}")
        print(f"     Вызовов:       {len(self.calls)}")
        print()
        print(f"  По шагам:")
        for call in self.calls:
            print(f"     {call['name']:20s} "
                  f"{call['prompt_tokens']:>5} вх. + {call['completion_tokens']:>5} вых. = "
                  f"{call['total_tokens']:>5} токенов, ${call['cost_usd']:.6f}")
    
    def reset(self):
        """Сбросить статистику"""
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.calls = []
        self._last_call = None


# Глобальный экземпляр трекера
_COST_TRACKER = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Вернуть глобальный трекер стоимости"""
    return _COST_TRACKER


def reset_cost_tracker():
    """Сбросить глобальный трекер"""
    _COST_TRACKER.reset()


# ============================================================================
# Фабрика клиента
# ============================================================================

def _make_openai_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_AUTH_TOKEN не задан. Либо экспортируй токен, "
                "либо положи LLM_AUTH_TOKEN=... в .env."
            )
        timeout = float(os.environ.get("LLM_TIMEOUT", "200"))
        http = httpx.Client(verify=False, timeout=timeout)
        return OpenAI(api_key=key, base_url=base, http_client=http)

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Ни LLM_BASE_URL, ни OPENAI_API_KEY не заданы. "
            "Сконфигурируй стенд через .env (см. .env.example)."
        )
    return OpenAI(api_key=key)


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


# ============================================================================
# JSON-парсинг из грязного ответа LLM
# ============================================================================

_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def _thinking_off_payload() -> dict:
    if os.environ.get("LLM_THINKING", "off").lower() in ("on", "1", "true", "yes"):
        return {}
    return {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "reasoning_effort": "none",
    }


def _clean(text: str) -> str:
    text = _HARMONY_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json(text: str):
    t = _clean(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"В ответе не найдено валидного JSON: {text[:300]!r}")


# ============================================================================
# Drop-in обёртка с автоматическим трекингом стоимости
# ============================================================================

class _Completions:
    def __init__(self, client: OpenAI):
        self._c = client
        self._call_name = "unknown"
    
    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T] | None = None,
        max_retries: int = 1,
        temperature: float = 0.0,
        call_name: str = "unknown",  # <-- НОВЫЙ ПАРАМЕТР: имя вызова для трекинга
        **kw: Any,
    ) -> T:
        """
        Вызов LLM с автоматическим трекингом стоимости.
        
        Args:
            call_name: имя шага (planner, critic, answer_generation)
            response_model: Pydantic модель для парсинга ответа
            ... остальные параметры как у OpenAI
            
        Returns:
            Распарсенный объект response_model (или сырой ответ, если response_model=None)
        """
        # Если response_model не указан — возвращаем сырой ответ
        if response_model is None:
            return self._create_raw(model, messages, temperature, call_name, **kw)
        
        # Автоматический трекинг стоимости
        tracker = get_cost_tracker()
        model_name = model or get_model()
        
        # list[Model] → оборачиваем в {items: [...]}
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array", "items": item_schema}},
                "required": ["items"],
            }
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        addendum = (
            f"\n\nОтвечай ОДНИМ валидным JSON-объектом по схеме:\n{schema_str}\n"
            "ТОЛЬКО JSON. Никакого текста до/после, никакого markdown, "
            "никаких комментариев, никаких повторных объектов."
        )
        if wrap_list:
            addendum += " Массив верни в поле `items`."

        msgs = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
        if sys_i is not None:
            msgs[sys_i]["content"] = msgs[sys_i]["content"] + addendum
        else:
            msgs.insert(0, {"role": "system", "content": addendum.lstrip()})

        thinking_kw = _thinking_off_payload()

        def _call(kw: dict):
            try:
                return self._c.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **kw,
                )
            except TypeError:
                safe = {k: v for k, v in kw.items() if k != "reasoning_effort"}
                return self._c.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **safe,
                )

        last_err: Exception | None = None
        raw: str = ""
        for attempt in range(max_retries + 1):
            try:
                try:
                    resp = _call(thinking_kw)
                except Exception as sdk_err:
                    msg = str(sdk_err)
                    bad = "reasoning_effort" in msg or "chat_template_kwargs" in msg or "enable_thinking" in msg
                    if bad and thinking_kw:
                        thinking_kw = {}
                        resp = _call(thinking_kw)
                    else:
                        raise
                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                result = adapter.validate_python(obj)
                
                # <-- АВТОМАТИЧЕСКИЙ ТРЕКИНГ СТОИМОСТИ
                usage = getattr(resp, 'usage', None)
                if usage is not None:
                    tracker.add_call(
                        name=call_name,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        model=model_name,
                        extra={
                            "temperature": temperature,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                        }
                    )
                
                return result
                
            except Exception as e:
                last_err = e
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({
                    "role": "user",
                    "content": f"Невалидный ответ: {e}. Верни ТОЛЬКО один корректный JSON по схеме.",
                })
        
        assert last_err is not None
        raise last_err
    
    def _create_raw(self, model: str, messages: list[dict], temperature: float, call_name: str, **kw):
        """Сырой вызов без JSON-парсинга (для агента)"""
        tracker = get_cost_tracker()
        model_name = model or get_model()
        
        thinking_kw = _thinking_off_payload()
        
        def _call(kw: dict):
            try:
                return self._c.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    **kw,
                )
            except TypeError:
                safe = {k: v for k, v in kw.items() if k != "reasoning_effort"}
                return self._c.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    **safe,
                )
        
        try:
            resp = _call(thinking_kw)
        except Exception as e:
            msg = str(e)
            bad = "reasoning_effort" in msg or "chat_template_kwargs" in msg or "enable_thinking" in msg
            if bad and thinking_kw:
                resp = _call({})
            else:
                raise
        
        # Автоматический трекинг
        usage = getattr(resp, 'usage', None)
        if usage is not None:
            tracker.add_call(
                name=call_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                model=model_name,
                extra={"temperature": temperature}
            )
        
        return resp


class _Chat:
    def __init__(self, client: OpenAI):
        self.completions = _Completions(client)


class JsonClient:
    """Клиент с автоматическим трекингом стоимости."""
    
    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _Chat(openai_client)


def make_client() -> JsonClient:
    """Вернуть клиент с API и автоматическим трекингом."""
    return JsonClient(_make_openai_client())


# ============================================================================
# «Сырой» клиент без JSON-инструктора
# ============================================================================

class _RawCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, call_name: str = "raw", **kw: Any):
        """Сырой вызов с автоматическим трекингом"""
        tracker = get_cost_tracker()
        model_name = kw.get('model', get_model())
        
        thinking = _thinking_off_payload()

        def _call(extra: dict):
            try:
                return self._inner.create(**kw, **extra)
            except TypeError:
                safe = {k: v for k, v in extra.items() if k != "reasoning_effort"}
                return self._inner.create(**kw, **safe)

        try:
            resp = _call(thinking)
        except Exception as e:
            msg = str(e)
            bad = "reasoning_effort" in msg or "chat_template_kwargs" in msg or "enable_thinking" in msg
            if bad and thinking:
                resp = _call({})
            else:
                raise
        
        usage = getattr(resp, 'usage', None)
        if usage is not None:
            tracker.add_call(
                name=call_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                model=model_name,
                extra={"temperature": kw.get('temperature', 0.0)}
            )
        
        return resp


class _RawChat:
    def __init__(self, inner):
        self.completions = _RawCompletions(inner.completions)


class RawClient:
    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _RawChat(openai_client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


def make_raw_client() -> RawClient:
    return RawClient(_make_openai_client())


# ============================================================================
# Упрощённый API с автоматическим трекингом
# ============================================================================

def ask_llm(
    prompt: str,
    response_model: Type[T] | None = None,
    call_name: str = "llm_call",
    temperature: float = 0.0,
    max_retries: int = 2,
    **kwargs
) -> T:
    """
    Упрощённый вызов LLM с автоматическим трекингом.
    
    Пример:
        answer = ask_llm(
            "Привет!",
            response_model=MovieAnswer,
            call_name="greeting"
        )
    """
    client = make_client()
    messages = [{"role": "user", "content": prompt}]
    
    if "system" in kwargs:
        messages.insert(0, {"role": "system", "content": kwargs.pop("system")})
    
    return client.chat.completions.create(
        model=kwargs.get("model", get_model()),
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_retries=max_retries,
        call_name=call_name,
        **kwargs
    )