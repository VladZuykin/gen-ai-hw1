# hw/discover.py
"""
Round 2.5 — Autodiscovery for app reviews (Spotify)
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import make_client, get_model
from prompts import DISCOVER_SYSTEM
from schema import DiscoveredAspects, AppAspects

client = make_client()
MODEL = get_model()


def discover_aspects(transcript: str) -> DiscoveredAspects:
    """Stage A: Discover topics from reviews"""
    
    # Take first 10 reviews to avoid overload
    chunks = transcript.split("\n\n---\n\n")
    sample = "\n\n---\n\n".join(chunks[:10])
    
    messages = [
        {"role": "system", "content": DISCOVER_SYSTEM},
        {"role": "user", "content": f"Extract 5-8 key topics from these app reviews:\n\n{sample}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=DiscoveredAspects,
        max_retries=3,
        temperature=0.3
    )
    
    return result


def extract_with_discovered(transcript: str, discovered: DiscoveredAspects) -> list[AppAspects]:
    """Stage B: Classify using discovered topics"""
    
    # Build topic list for prompt
    aspects_list = "\n".join([
        f"- {a.name}: {a.description}" 
        for a in discovered.aspects
    ])
    
    dynamic_prompt = f"""You are a sentiment analyzer for app reviews.

DISCOVERED TOPICS (use ONLY these):
{aspects_list}

TASK:
For each review, identify which topics from the list above are mentioned and with what sentiment.

RULES:
1. Use ONLY topics from the list above
2. Each sentiment MUST be supported by a VERBATIM quote
3. Sentiment: positive / neutral / negative

OUTPUT FORMAT:
Return list of reviews with fields: app_name, aspects (list with aspect, sentiment, quote)"""

    messages = [
        {"role": "system", "content": dynamic_prompt},
        {"role": "user", "content": f"Analyze reviews by discovered topics:\n\n{transcript}"}
    ]
    
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=list[AppAspects],
        max_retries=3,
        temperature=0.0
    )
    
    return result


def main():
    """Run autodiscovery"""
    
    # Read reviews
    input_dir = Path("input")
    if not input_dir.exists():
        print("❌ Input folder not found")
        return
    
    texts = []
    for f in sorted(input_dir.glob("*.txt"))[:20]:  # 20 reviews for speed
        texts.append(f.read_text(encoding="utf-8"))
    transcript = "\n\n---\n\n".join(texts)
    
    print("━━━ Stage A: Discovering topics ━━━")
    discovered = discover_aspects(transcript)
    print(f"Found {len(discovered.aspects)} topics:")
    for a in discovered.aspects:
        print(f"  • {a.name} — {a.description}")
    
    print("\n━━━ Stage B: Classifying by discovered topics ━━━")
    aspects = extract_with_discovered(transcript, discovered)
    print(f"Processed reviews: {len(aspects)}")
    
    # Compare with fixed aspects
    fixed_path = Path("output/aspects.json")
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
        fixed_aspects = {a["aspect"] for item in fixed for a in item.get("aspects", [])}
        dyn_aspects = {a.aspect for item in aspects for a in item.aspects}
        
        new = dyn_aspects - fixed_aspects
        missing = fixed_aspects - dyn_aspects
        
        print(f"\n━━━ Comparison with fixed aspects (output/aspects.json) ━━━")
        print(f"  Fixed aspects: {len(fixed_aspects)} → {sorted(fixed_aspects)}")
        print(f"  Discovered:    {len(dyn_aspects)} → {sorted(dyn_aspects)}")
        
        if new:
            print(f"\n  ⊕ NEW topics (not in fixed list):")
            for t in sorted(new):
                print(f"    • {t}")
        
        if missing:
            print(f"\n  ⊖ MISSING (fixed topics not found in reviews):")
            for t in sorted(missing):
                print(f"    • {t}")
    else:
        print("\n⚠️ No output/aspects.json for comparison")
    
    # Save results
    out_path = Path("output_discovered")
    out_path.mkdir(exist_ok=True)
    
    with open(out_path / "aspects_discovered.json", "w", encoding="utf-8") as f:
        json.dump([a.model_dump() for a in aspects], f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved: {out_path}/aspects_discovered.json")


if __name__ == "__main__":
    main()