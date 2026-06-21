"""
Утилиты: логирование.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TraceLogger:
    """Логирование всех шагов пайплайна"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace = []
        self.trace_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_query(self, query):
        self.trace.append({
            "step": "query",
            "timestamp": datetime.now().isoformat(),
            "query": query.model_dump() if hasattr(query, 'model_dump') else query
        })
        
    def log_plan(self, plan):
        self.trace.append({
            "step": "plan",
            "timestamp": datetime.now().isoformat(),
            "plan": plan.model_dump() if hasattr(plan, 'model_dump') else plan
        })
        
    def log_search_results(self, rag_results, web_results):
        self.trace.append({
            "step": "search",
            "timestamp": datetime.now().isoformat(),
            "rag_count": len(rag_results),
            "web_count": len(web_results)
        })
        
    def log_answer(self, answer):
        self.trace.append({
            "step": "answer",
            "timestamp": datetime.now().isoformat(),
            "answer": answer.model_dump() if hasattr(answer, 'model_dump') else answer
        })
        
    def log_verdict(self, verdict):
        self.trace.append({
            "step": "verdict",
            "timestamp": datetime.now().isoformat(),
            "verdict": verdict.model_dump() if hasattr(verdict, 'model_dump') else verdict
        })
        
    def save(self):
        """Сохранить трейс в файл"""
        trace_file = self.output_dir / f"trace_{self.trace_id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, ensure_ascii=False, indent=2)
        return trace_file