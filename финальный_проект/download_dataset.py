"""
Скрипт для скачивания датасета с Kaggle.
Запуск: python download_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd
import kagglehub

from config import DATA_DIR


def download_movie_dataset() -> Path:
    """
    Скачать датасет фильмов с Kaggle в папку input/.
    
    Returns:
        Path к сохранённому CSV-файлу
    """
    print("=" * 60)
    print("📥 СКАЧИВАНИЕ ДАТАСЕТА С KAGGLE")
    print("=" * 60)
    
    dataset_name = "maksimpotorochin/movie-plots-from-wikipedia-in-russian"
    print(f"\n📦 Датасет: {dataset_name}")
    
    try:
        # Скачиваем через kagglehub
        print("⏳ Загрузка...")
        dataset_path = kagglehub.dataset_download(dataset_name)
        print(f"✅ Скачано в: {dataset_path}")
        
        # Ищем CSV-файл
        csv_files = list(Path(dataset_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("CSV-файл не найден в скачанном датасете")
        
        csv_source = csv_files[0]
        print(f"📄 Найден файл: {csv_source.name}")
        
        # Читаем и показываем превью
        df = pd.read_csv(csv_source)
        print(f"\n📊 Статистика:")
        print(f"   Строк: {len(df)}")
        print(f"   Колонки: {', '.join(df.columns)}")
        
        # Проверяем наличие сюжетов
        plot_col = None
        for col in ['plot', 'Plot', 'description', 'Description', 'text']:
            if col in df.columns:
                plot_col = col
                break
        
        if plot_col:
            non_empty = df[plot_col].notna().sum()
            print(f"   Сюжеты: {non_empty} / {len(df)} ({(non_empty/len(df)*100):.1f}%)")
        else:
            print("   ⚠️ Колонка с сюжетами не найдена!")
            print(f"   Доступные колонки: {list(df.columns)}")
        
        # Сохраняем в папку input/
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DATA_DIR / "movies.csv"
        df.to_csv(target_path, index=False, encoding="utf-8")
        print(f"\n💾 Сохранено: {target_path}")
        
        # Показываем пример
        print("\n📋 Пример первых строк:")
        print("=" * 60)
        if plot_col:
            sample = df[[plot_col]].head(3)
            for i, row in sample.iterrows():
                text = str(row[plot_col])[:200] + "..." if len(str(row[plot_col])) > 200 else str(row[plot_col])
                print(f"\n[{i+1}] {text}")
        else:
            print(df.head(3).to_string())
        
        return target_path
        
    except ImportError:
        print("\n❌ Ошибка: kagglehub не установлен")
        print("   Установите: pip install kagglehub")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Альтернатива: скачайте датасет вручную с Kaggle:")
        print(f"   https://www.kaggle.com/datasets/{dataset_name}")
        print("   и положите CSV в папку input/")
        sys.exit(1)


def main():
    """Запуск скачивания"""
    csv_path = download_movie_dataset()
    
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print(f"   Датасет сохранён: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()