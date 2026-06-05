"""
Финал — Сборка конвейера
==========================
Всё, что мы разложили по раундам 1-7, теперь собираем в один analyze().
Никаких новых концепций — только связывание. На входе путь к транскрипту,
на выходе папка output/ со всеми артефактами.

Задача:
  Дописать analyze(transcript_path, out_dir). Запустить, проверить,
  что в out_dir/ появилось:
    • participants.json + participants.csv
    • aspects.json + heatmap.png
    • summary.json
    • judge_report.json
    • metrics.json (полнота/точность/достоверность)

Запуск:
    python 12_pipeline.py transcript.txt
    python 12_pipeline.py transcripts/bank_olimp.txt output/olimp
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_p = importlib.import_module("2_extract_participants")
extract_participants = _p.extract_participants

_a = importlib.import_module("4_extract_aspects")
extract_aspects = _a.extract_aspects
check_quotes = _a.check_quotes
build_heatmap = _a.build_heatmap

_mr = importlib.import_module("7_map_reduce")
summarize_discussion = _mr.summarize_discussion

_j = importlib.import_module("9_judge")
judge = _j.judge

_eval = importlib.import_module("3_evaluate_ie")
fidelity = _eval.fidelity
coverage = _eval.coverage


def analyze(transcript_path: str, out_dir: str = "output") -> None:
    """Полный конвейер: транскрипт → набор артефактов в out_dir/."""

    # Создаём папку для результатов
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Читаем транскрипт
    transcript = Path(transcript_path).read_text(encoding="utf-8")
    print(f"\n📄 Анализ: {transcript_path} ({len(transcript)} символов)")
    
    # ===== 1. Извлечение участников =====
    print("\n━━━ Раунд 1: Извлечение участников ━━━")
    participants = extract_participants(transcript)
    
    # Сохраняем JSON
    participants_json = out_path / "participants.json"
    participants_json.write_text(
        json.dumps([p.model_dump() for p in participants], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  ✅ Найдено {len(participants)} участников")
    
    # Сохраняем CSV для Excel
    participants_csv = out_path / "participants.csv"
    import csv
    with open(participants_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age", "city", "profession", "concern_category", "concern_text", "quote"])
        for p in participants:
            for concern in p.concerns:
                writer.writerow([p.name, p.age, p.city, p.profession, 
                               concern.category, concern.text, concern.quote])
    print(f"  ✅ CSV: {participants_csv}")
    
    # ===== 2. Аспектный анализ =====
    print("\n━━━ Раунд 2: Аспектный анализ ━━━")
    aspects = extract_aspects(transcript)
    
    # Проверяем цитаты
    ghosts = check_quotes(aspects, transcript)
    if ghosts:
        print(f"  ⚠ Найдено {len(ghosts)} проблемных цитат (возможно, галлюцинации)")
        for name, quote in ghosts[:3]:
            print(f"    - {name}: {quote[:60]}...")
    else:
        print(f"  ✅ Все цитаты валидны")
    
    # Сохраняем aspects.json
    aspects_json = out_path / "aspects.json"
    aspects_json.write_text(
        json.dumps([p.model_dump() for p in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # Строим тепловую карту
    heatmap_path = out_path / "heatmap.png"
    build_heatmap(aspects, str(heatmap_path))
    print(f"  ✅ Тепловая карта: {heatmap_path}")
    
    # ===== 3. Map-Reduce резюме =====
    print("\n━━━ Раунд 3: Map-Reduce резюме ━━━")
    summary = summarize_discussion(transcript)
    
    summary_json = out_path / "summary.json"
    summary_json.write_text(
        summary.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  ✅ Заголовок: {summary.headline[:80]}...")
    print(f"  ✅ {len(summary.key_findings)} выводов, {len(summary.action_items)} рекомендаций")
    
    # ===== 4. Оценка судьи (если есть эталон) =====
    print("\n━━━ Раунд 5: LLM-as-Judge ━━━")
    
    # Загружаем participants.json для судьи
    participants_data = json.loads(participants_json.read_text(encoding="utf-8"))
    summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
    
    judge_report = judge(participants_data, summary_data)
    
    judge_json = out_path / "judge_report.json"
    judge_json.write_text(
        judge_report.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  ✅ Overall score: {judge_report.overall_score:.2f}")
    print(f"  ✅ Supported: {sum(1 for v in judge_report.verdicts if v.support == 'supported')}/{len(judge_report.verdicts)}")
    
    # ===== 5. Метрики (если есть baseline) =====
    baseline_path = Path("baseline_manual.json")
    if baseline_path.exists():
        print("\n━━━ Метрики качества (сравнение с эталоном) ━━━")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        
        f = fidelity(participants_data, transcript)
        c = coverage(baseline, participants_data)
        
        metrics = {
            "fidelity": f,
            "coverage": c,
            "judge_score": judge_report.overall_score,
            "total_participants": len(participants),
            "total_aspects": sum(len(p.aspects) for p in aspects),
            "total_recommendations": len(summary.action_items)
        }
        
        metrics_json = out_path / "metrics.json"
        metrics_json.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        print(f"  ✅ Достоверность цитат: {f:.1%}")
        print(f"  ✅ Полнота (vs эталон): {c:.1%}")
        print(f"  ✅ Сохранено: {metrics_json}")
    
    # ===== Финальный вывод =====
    print("\n" + "="*50)
    print(f"✅ Пайплайн завершён! Результаты в папке: {out_dir}/")
    print("="*50)
    print(f"\n📁 {participants_json}")
    print(f"📁 {participants_csv}")
    print(f"📁 {aspects_json}")
    print(f"📁 {heatmap_path}")
    print(f"📁 {summary_json}")
    print(f"📁 {judge_json}")
    if baseline_path.exists():
        print(f"📁 {out_path}/metrics.json")


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python 12_pipeline.py <transcript.txt> [out_dir]")
        sys.exit(1)
    transcript_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    analyze(transcript_path, out_dir)


if __name__ == "__main__":
    main()
