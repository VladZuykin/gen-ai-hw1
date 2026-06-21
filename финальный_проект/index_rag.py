"""
Скрипт для индексации датасета в RAG.
Запуск: python index_rag.py [--max 5000] [--force]
"""

import sys
import time
from pathlib import Path
from typing import Optional  # <-- ДОБАВЛЯЕМ ИМПОРТ

from rag import MovieRAG
from config import DATA_DIR


def index_rag(
    max_films: Optional[int] = None,
    force_reindex: bool = False,
    use_summary: bool = True,
):
    """
    Индексировать датасет фильмов в RAG.
    
    Args:
        max_films: максимальное количество фильмов (None = все)
        force_reindex: переиндексировать даже если кэш существует
        use_summary: добавлять summary к plot
    """
    print("=" * 60)
    print("🔧 ИНДЕКСАЦИЯ RAG")
    print("=" * 60)
    
    csv_path = DATA_DIR / "movies.csv"
    if not csv_path.exists():
        print("\n❌ Датасет не найден!")
        print("   Сначала запусти: python download_dataset.py")
        sys.exit(1)
    
    print(f"\n📄 Датасет: {csv_path}")
    print(f"   Размер: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    if max_films:
        print(f"   Ограничение: {max_films} фильмов")
    print(f"   Summary: {'включён' if use_summary else 'выключен'}")
    
    # Проверяем кэш
    cache_path = DATA_DIR / "rag_cache.json"
    if cache_path.exists() and not force_reindex:
        cache_size = cache_path.stat().st_size / 1024 / 1024
        print(f"\n📦 Кэш уже существует: {cache_path} ({cache_size:.1f} MB)")
        response = input("   Переиндексировать? (y/N): ")
        if response.lower() != 'y':
            print("   ⏭️ Пропускаем")
            return
    
    print("\n⏳ Индексация... (это может занять несколько минут)")
    start_time = time.time()
    
    try:
        rag = MovieRAG()
        rag.load_from_csv(
            csv_path,
            max_films=max_films,
            use_summary=use_summary,
        )
        
        rag.save_cache(cache_path)
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 Результат:")
        print(f"   Чанков: {len(rag.chunks)}")
        print(f"   Время: {elapsed:.1f}с")
        print(f"   Кэш сохранён: {cache_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Индексация прервана пользователем")
        print("   Попробуйте с меньшим количеством фильмов:")
        print("   python index_rag.py --max 5000")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Ошибка индексации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("   RAG проиндексирован и готов к использованию")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Индексация RAG")
    parser.add_argument(
        "--max", 
        type=int, 
        default=None,
        help="Максимальное количество фильмов для индексации"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Переиндексировать даже если кэш существует"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Не использовать summary (только plot)"
    )
    args = parser.parse_args()
    
    index_rag(
        max_films=args.max,
        force_reindex=args.force,
        use_summary=not args.no_summary,
    )


if __name__ == "__main__":
    main()