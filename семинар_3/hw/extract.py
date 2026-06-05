# hw/extract.py
from __future__ import annotations

from llm_client import make_client, get_model
from schema import Review
from prompts import IE_SYSTEM

client = make_client()
MODEL = get_model()


def extract_reviews_with_usage(transcript: str) -> tuple[list[Review], dict]:
    """Extract reviews with usage tracking"""
    messages = [
        {"role": "system", "content": IE_SYSTEM},
        {"role": "user", "content": f"Analyze these app reviews:\n\n{transcript}"}
    ]
    
    result, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=list[Review],
        max_retries=3,
        temperature=0.0,
        with_completion=True
    )
    
    usage = {
        'prompt_tokens': completion.usage.prompt_tokens,
        'completion_tokens': completion.usage.completion_tokens,
        'total_tokens': completion.usage.total_tokens
    }
    
    return result, usage


def extract_reviews(transcript: str) -> list[Review]:
    result, _ = extract_reviews_with_usage(transcript)
    return result