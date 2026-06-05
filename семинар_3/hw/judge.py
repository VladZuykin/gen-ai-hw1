# hw/judge.py
from __future__ import annotations

import json
from pathlib import Path

from llm_client import make_client, get_model
from prompts import JUDGE_SYSTEM
from schema import JudgeReport

client = make_client()
MODEL = get_model()


def build_evidence_packet(reviews_data: list[dict], summary_data: dict) -> str:
    """Build evidence packet for the judge"""
    
    # Format reviews (first 15 for context)
    reviews_text = "## Original reviews (facts):\n\n"
    for review in reviews_data[:15]:
        # Use app_name instead of anime_title for Spotify
        reviews_text += f"### {review.get('app_name', review.get('anime_title', 'Unknown'))}\n"
        reviews_text += f"Rating: {review.get('rating', '?')}/5\n"
        reviews_text += f"Sentiment: {review.get('sentiment', '?')}\n"
        
        # Handle both 'likes' (Spotify) and 'praises' (anime)
        likes = review.get('likes', review.get('praises', []))
        if likes:
            reviews_text += "Likes:\n"
            for l in likes[:2]:
                reviews_text += f"  - {l.get('text', '')[:100]}\n"
        
        # Handle both 'issues' (Spotify) and 'critiques' (anime)
        issues = review.get('issues', review.get('critiques', []))
        if issues:
            reviews_text += "Issues:\n"
            for i in issues[:2]:
                reviews_text += f"  - {i.get('text', '')[:100]}\n"
        reviews_text += "\n"
    
    # Format recommendations
    recommendations_text = "## Recommendations to verify:\n\n"
    for i, rec in enumerate(summary_data.get('recommendations', []), 1):
        recommendations_text += f"{i}. {rec}\n"
    
    return reviews_text + "\n" + recommendations_text


def judge_reviews_with_usage(reviews_data: list[dict], summary_data: dict) -> tuple[JudgeReport, dict]:
    """Judge with usage tracking"""
    evidence = build_evidence_packet(reviews_data, summary_data)
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Check these recommendations against the reviews:\n\n{evidence}"}
    ]
    
    result, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=JudgeReport,
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


def judge_reviews(reviews_data: list[dict], summary_data: dict) -> JudgeReport:
    result, _ = judge_reviews_with_usage(reviews_data, summary_data)
    return result


def main():
    """Run the judge"""
    reviews_path = Path("output/reviews.json")
    summary_path = Path("output/summary.json")
    
    if not reviews_path.exists():
        reviews_path = Path("output_anime/reviews.json")
        summary_path = Path("output_anime/summary.json")
    
    if not reviews_path.exists():
        print("❌ reviews.json not found. Run pipeline.py first")
        return
    
    print("━━━ LLM-as-Judge: Quality Assessment ━━━\n")
    
    reviews_data = json.loads(reviews_path.read_text(encoding="utf-8"))
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    
    report = judge_reviews(reviews_data, summary_data)
    
    out_path = reviews_path.parent / "judge_report.json"
    out_path.write_text(
        report.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"✅ Overall score: {report.overall_score:.2f}")
    print(f"✅ Verdicts: {len(report.verdicts)}")
    
    for v in report.verdicts:
        mark = {"supported": "✓", "weakly_supported": "?", "not_supported": "✗"}[v.support]
        print(f"  {mark} [{v.support}] {v.action[:60]}...")
    
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()