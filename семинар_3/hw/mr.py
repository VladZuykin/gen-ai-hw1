# hw/mr.py
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import make_client, get_model
from prompts import CHUNK_SYSTEM, REDUCE_SYSTEM
from schema import ReviewSummary, AggregateReport

client = make_client()
MODEL = get_model()


def split_reviews(text: str) -> list[str]:
    """Split reviews by separator ---"""
    return [chunk.strip() for chunk in text.split("---") if chunk.strip()]


def summarize_chunk_with_usage(chunk: str) -> tuple[ReviewSummary, dict]:
    """MAP with usage tracking"""
    messages = [
        {"role": "system", "content": CHUNK_SYSTEM},
        {"role": "user", "content": f"Summarize this review:\n\n{chunk}"}
    ]
    
    result, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=ReviewSummary,
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


def reduce_summaries_with_usage(summaries: list[ReviewSummary]) -> tuple[AggregateReport, dict]:
    """REDUCE with usage tracking"""
    summaries_text = []
    for s in summaries:
        summaries_text.append(f"## {s.app_name} ({s.sentiment})")
        summaries_text.append(f"Rating: {s.rating}/5")
        summaries_text.append(f"Key points: {', '.join(s.key_points)}")
        summaries_text.append("")
    
    combined = "\n".join(summaries_text)
    
    messages = [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content": f"Create a summary report from these reviews:\n\n{combined}"}
    ]
    
    result, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=AggregateReport,
        max_retries=3,
        temperature=0.0,
        with_completion=True
    )
    
    usage = {
        'prompt_tokens': completion.usage.prompt_tokens,
        'completion_tokens': completion.usage.completion_tokens,
        'total_tokens': completion.usage.total_tokens,
        'cost_usd': (completion.usage.prompt_tokens * 0.14 + completion.usage.completion_tokens * 0.55) / 1_000_000
    }
    
    return result, usage


def summarize_reviews_with_usage(transcript: str, workers: int = 6) -> tuple[AggregateReport, dict]:
    """Full Map-Reduce pipeline with usage tracking"""
    chunks = split_reviews(transcript)
    n = len(chunks)
    print(f"  [MR] MAP: {n} reviews, up to {workers} parallel...")
    t0 = time.time()
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    summaries = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summarize_chunk_with_usage, c): i for i, c in enumerate(chunks)}
        
        for future in as_completed(futures):
            i = futures[future]
            summary, usage = future.result()
            summaries[i] = summary
            total_prompt_tokens += usage['prompt_tokens']
            total_completion_tokens += usage['completion_tokens']
    
    t_map = time.time() - t0
    
    cost_map = (total_prompt_tokens * 0.14 + total_completion_tokens * 0.55) / 1_000_000
    print(f"  [MR] MAP: {t_map:.1f}s, tokens: {total_prompt_tokens}+{total_completion_tokens}, ${cost_map:.4f}")
    
    print(f"  [MR] REDUCE...")
    result, reduce_usage = reduce_summaries_with_usage([s for s in summaries if s])
    
    total_prompt_tokens += reduce_usage['prompt_tokens']
    total_completion_tokens += reduce_usage['completion_tokens']
    total_cost = (total_prompt_tokens * 0.14 + total_completion_tokens * 0.55) / 1_000_000
    
    print(f"  [MR] total {time.time() - t0:.1f}s, total cost: ${total_cost:.4f}")
    
    total_usage = {
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'total_tokens': total_prompt_tokens + total_completion_tokens,
        'cost_usd': total_cost
    }
    
    return result, total_usage


def summarize_reviews(transcript: str, workers: int = 6) -> AggregateReport:
    result, _ = summarize_reviews_with_usage(transcript, workers)
    return result