# hw/pipeline.py
from __future__ import annotations

import json
from pathlib import Path

from extract import extract_reviews_with_usage
from aspects import analyze_aspects_with_usage, check_quotes, build_heatmap
from mr import summarize_reviews_with_usage
from judge import judge_reviews_with_usage


def analyze(input_path: str, out_dir: str = "output") -> None:
    """Full pipeline with cost tracking"""
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Read input data
    input_path_obj = Path(input_path)
    if input_path_obj.is_dir():
        texts = []
        for f in sorted(input_path_obj.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
        transcript = "\n\n---\n\n".join(texts)
        print(f"\n📂 Loaded {len(texts)} files from folder {input_path}")
    else:
        transcript = input_path_obj.read_text(encoding="utf-8")
        print(f"\n📄 Loaded file: {input_path}")
    
    print(f"   Total size: {len(transcript)} characters")
    
    # Counters
    total_cost = 0.0
    all_usages = []
    
    # ===== Round 1: IE =====
    print("\n━━━ Round 1: Information Extraction (IE) ━━━")
    reviews, usage_ie = extract_reviews_with_usage(transcript)
    all_usages.append(usage_ie)
    cost_ie = (usage_ie['prompt_tokens'] * 0.14 + usage_ie['completion_tokens'] * 0.55) / 1_000_000
    total_cost += cost_ie
    
    reviews_json = out_path / "reviews.json"
    reviews_json.write_text(
        json.dumps([r.model_dump() for r in reviews], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  ✅ Processed {len(reviews)} reviews (${cost_ie:.4f})")
    
    # ===== Round 2: Aspect-Based Analysis =====
    print("\n━━━ Round 2: Aspect-Based Analysis ━━━")
    aspects, usage_aspects = analyze_aspects_with_usage(transcript)
    all_usages.append(usage_aspects)
    cost_aspects = (usage_aspects['prompt_tokens'] * 0.14 + usage_aspects['completion_tokens'] * 0.55) / 1_000_000
    total_cost += cost_aspects

    # Check for ghost quotes (hallucinations)
    ghosts = check_quotes(aspects, transcript)
    if ghosts:
        print(f"  ⚠ Found {len(ghosts)} problematic quotes (hallucinations)")
        for name, quote in ghosts[:3]:
            print(f"    - {name}: {quote[:80]}...")
    else:
        print(f"  ✅ All quotes valid, ghost quotes: 0")

    aspects_json = out_path / "aspects.json"
    aspects_json.write_text(
        json.dumps([a.model_dump() for a in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  ✅ Saved (${cost_aspects:.4f})")

    heatmap_path = out_path / "heatmap.png"
    build_heatmap(aspects, str(heatmap_path))
    print(f"  ✅ Heatmap: {heatmap_path}")
    
    # ===== Round 3: Map-Reduce =====
    print("\n━━━ Round 3: Map-Reduce ━━━")
    summary, usage_mr = summarize_reviews_with_usage(transcript)
    all_usages.append(usage_mr)
    total_cost += usage_mr['cost_usd']
    
    summary_json = out_path / "summary.json"
    summary_json.write_text(
        summary.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  ✅ Headline: {summary.headline[:80]}...")
    
    # ===== Round 5: LLM-as-Judge =====
    print("\n━━━ Round 5: LLM-as-Judge ━━━")
    reviews_data = json.loads(reviews_json.read_text(encoding="utf-8"))
    summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
    
    judge_report, usage_judge = judge_reviews_with_usage(reviews_data, summary_data)
    all_usages.append(usage_judge)
    cost_judge = (usage_judge['prompt_tokens'] * 0.14 + usage_judge['completion_tokens'] * 0.55) / 1_000_000
    total_cost += cost_judge
    
    judge_json = out_path / "judge_report.json"
    judge_json.write_text(
        judge_report.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  ✅ Overall score: {judge_report.overall_score:.2f} (${cost_judge:.4f})")
    
    # ===== Final Statistics =====
    total_prompt = sum(u['prompt_tokens'] for u in all_usages)
    total_completion = sum(u['completion_tokens'] for u in all_usages)
    
    print("\n" + "="*60)
    print("💰 FINAL COST")
    print("="*60)
    print(f"   Total LLM calls: {len(all_usages)} (1 IE + 1 aspects + N MAP + 1 REDUCE + 1 Judge)")
    print(f"   Prompt tokens:     {total_prompt:,}")
    print(f"   Completion tokens: {total_completion:,}")
    print(f"   Total tokens:      {total_prompt + total_completion:,}")
    print(f"   💰 Total cost:      ${total_cost:.4f} (~{total_cost * 100:.2f} cents)")
    print("="*60)
    
    print(f"\n✅ Pipeline completed! Results in folder: {out_dir}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <input_dir or input.txt> [out_dir]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    analyze(input_path, out_dir)