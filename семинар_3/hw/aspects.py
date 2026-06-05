# hw/aspects.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from llm_client import make_client, get_model

from schema import AppAspects
from prompts import ASPECTS_SYSTEM

client = make_client()
MODEL = get_model()


def analyze_aspects_with_usage(transcript: str) -> tuple[list[AppAspects], dict]:
    """Aspect-based analysis with usage tracking"""
    messages = [
        {"role": "system", "content": ASPECTS_SYSTEM},
        {"role": "user", "content": f"Analyze these app reviews by aspects:\n\n{transcript}"}
    ]
    
    result, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=list[AppAspects],
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


def analyze_aspects(transcript: str) -> list[AppAspects]:
    result, _ = analyze_aspects_with_usage(transcript)
    return result


def check_quotes(aspects: list[AppAspects], transcript: str) -> list[tuple[str, str]]:
    """Check for hallucinated quotes (not found in original text)"""
    ghosts = []
    transcript_lower = transcript.lower()
    
    for a in aspects:
        for asp in a.aspects:
            quote = asp.quote
            if quote and quote.lower() not in transcript_lower:
                ghosts.append((a.app_name, quote[:80]))
    
    return ghosts


def build_heatmap(aspects: list[AppAspects], out_path: str = "heatmap.png") -> None:
    """Build heatmap for Spotify reviews"""
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    
    apps = [a.app_name[:30] for a in aspects[:10]]
    all_aspects = ["performance", "design", "support", "price", "ads", "reliability"]
    
    matrix = np.full((len(apps), len(all_aspects)), np.nan)
    
    for i, a in enumerate(aspects[:10]):
        for asp in a.aspects:
            if asp.aspect in all_aspects:
                j = all_aspects.index(asp.aspect)
                matrix[i, j] = sentiment_map.get(asp.sentiment, 0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, 
                xticklabels=all_aspects, 
                yticklabels=apps, 
                annot=True, 
                fmt='.0f', 
                cmap="RdYlGn", 
                center=0,
                cbar_kws={'label': 'Positive=1, Neutral=0, Negative=-1'})
    plt.title("Aspect-Based Sentiment Analysis for Spotify Reviews")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Heatmap saved: {out_path}")