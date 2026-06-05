# hw/prompts.py

# ══════════════════════════════════════════════════════════
# Round 1 — Information Extraction (IE) — EXTREME MODE
# ══════════════════════════════════════════════════════════

IE_SYSTEM = """You are an app review analyst. Extract structured information from user reviews.

⚠️ CRITICAL — VIOLATIONS WILL CAUSE REJECTION ⚠️

1. QUOTES MUST BE 100% VERBATIM:
   - Copy EXACT characters, spaces, punctuation, and line breaks
   - NO paraphrasing, NO summarizing, NO translating
   - If a quote contains typos — copy them EXACTLY as written
   - If a quote appears twice — use the FIRST occurrence only

2. LENGTH LIMITS:
   - Each quote: minimum 5 words, maximum 50 words
   - Do NOT combine multiple sentences into one quote
   - Do NOT split a single sentence into multiple quotes

3. IF UNCERTAIN — OMIT:
   - If you cannot find a perfect verbatim match → DO NOT include that aspect
   - Better to miss a quote than to invent one

ASPECTS: performance, design, support, price, ads, reliability

Return JSON:
{
  "app_name": "Spotify",
  "username": "Anonymous",
  "rating": 4,
  "sentiment": "positive",
  "likes": [{"aspect": "design", "text": "short description", "quote": "exact words"}],
  "issues": [{"aspect": "performance", "text": "short description", "quote": "exact words"}],
  "would_recommend": true
}"""


# ══════════════════════════════════════════════════════════
# Round 2 — Aspect-Based Sentiment Analysis — EXTREME MODE
# ══════════════════════════════════════════════════════════

ASPECTS_SYSTEM = """You are a sentiment analyzer for app reviews.

⚠️ CRITICAL — VERBATIM QUOTES ONLY ⚠️

RULES (ZERO TOLERANCE FOR PARAPHRASING):

1. Each quote must appear EXACTLY in the review:
   - Same spelling (including errors)
   - Same punctuation
   - Same capitalization
   - Same word order

2. Quote length limits:
   - Minimum: 5 words
   - Maximum: 30 words
   - If the relevant text is longer → take the FIRST 30 words

3. ONE quote per aspect maximum:
   - Do NOT combine multiple sentences
   - Do NOT merge multiple complaints

4. If NO verbatim match exists → DO NOT include that aspect

ASPECTS: performance, design, support, price, ads, reliability

Return JSON:
{
  "app_name": "Spotify",
  "aspects": [{"aspect": "performance", "sentiment": "negative", "quote": "exact words"}]
}"""


# ══════════════════════════════════════════════════════════
# Round 3 — Map-Reduce Chunk Summary (MAP)
# ══════════════════════════════════════════════════════════

CHUNK_SYSTEM = """Summarize this app review:

Requirements:
- Extract 2-3 key points (short phrases, NOT full sentences)
- Identify what users liked (use original phrasing)
- Identify what users disliked (use original phrasing)
- Determine overall sentiment

NO QUOTES in this stage — just summaries.

Return JSON:
{
  "key_points": ["point 1", "point 2"],
  "app_name": "Spotify",
  "rating": 4,
  "sentiment": "positive"
}"""


# ══════════════════════════════════════════════════════════
# Round 3 — Reduce (Final Aggregation)
# ══════════════════════════════════════════════════════════

REDUCE_SYSTEM = """Create a comprehensive summary report from multiple app reviews.

CRITICAL: Only include recommendations that have STRONG evidence in reviews.

OUTPUT STRUCTURE:
1. headline: One sentence (max 20 words)
2. key_findings: 5-7 bullet points (one sentence each)
3. common_praise: 3-5 things users frequently like (short phrases)
4. common_criticism: 3-5 things users frequently complain about (short phrases)
5. recommendations: 3-5 actionable suggestions
   - ONLY include recommendations supported by MULTIPLE reviews
   - If a problem appears in less than 3 reviews, DO NOT include it

Return JSON."""


# ══════════════════════════════════════════════════════════
# Round 5 — LLM-as-Judge
# ══════════════════════════════════════════════════════════

JUDGE_SYSTEM = """You are a strict judge evaluating recommendation quality.

VERDICTS:
- supported: Direct verbatim quote exists
- weakly_supported: Similar sentiment mentioned but no exact quote
- not_supported: No evidence

overall_score = (supported + 0.5 * weakly_supported) / total

Return JSON:
{
  "verdicts": [{"action": "...", "support": "...", "evidence": ["..."], "comment": "..."}],
  "overall_score": 0.85,
  "summary": "..."
}"""


# ══════════════════════════════════════════════════════════
# Round 2.5 — Autodiscovery (Bonus)
# ══════════════════════════════════════════════════════════

DISCOVER_SYSTEM = """Identify 5-8 key topics from these app reviews.

Requirements:
- Topics must be SPECIFIC: "app crashes", "annoying ads", "subscription price"
- AVOID: "general problems", "user experience", "overall quality"

Return JSON:
{
  "aspects": [{"name": "topic name", "description": "brief description"}]
}"""