import pandas as pd
from pathlib import Path
import json
import re
from datetime import datetime

def load_spotify_data(csv_path: str = "reviews.csv") -> pd.DataFrame:
    """Загрузить Spotify отзывы"""
    
    print(f"Загрузка {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"✅ Загружено {len(df)} строк")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Удаляем пустые отзывы
    df = df[df['Review'].notna()]
    df = df[df['Review'].str.strip() != '']
    
    # Удаляем дубликаты
    df = df.drop_duplicates(subset=['Review'])
    
    # Добавляем дату для информации
    if 'Time_submitted' in df.columns:
        df['date'] = pd.to_datetime(df['Time_submitted'], errors='coerce')
    
    print(f"✅ После очистки: {len(df)} строк")
    
    return df

def show_stats(df: pd.DataFrame):
    """Показать статистику по датасету"""
    
    print(f"\n━━━ Статистика Spotify отзывов ━━━")
    print(f"  Всего отзывов: {len(df)}")
    print(f"  Средний рейтинг: {df['Rating'].mean():.2f}/5")
    
    # Распределение оценок
    rating_dist = df['Rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {rating}★: {count:>6} ({pct:>5.1f}%) {bar}")
    
    # Средняя длина отзыва
    avg_length = df['Review'].str.len().mean()
    print(f"  Средняя длина отзыва: {avg_length:.0f} символов")
    
    # Среднее количество лайков
    if 'Total_thumbsup' in df.columns:
        avg_likes = df['Total_thumbsup'].mean()
        print(f"  Среднее число лайков: {avg_likes:.1f}")

def clean_text(text: str) -> str:
    """Очистить текст от эмодзи и спецсимволов"""
    text = re.sub(r'[^\w\s\u0400-\u04FF.!?,-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def map_rating_to_sentiment(rating: int) -> str:
    """Преобразовать рейтинг 1-5 в тональность"""
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

def prepare_reviews(df: pd.DataFrame, n_samples: int = 50) -> list[dict]:
    """Подготовить отзывы для анализа"""
    
    reviews = []
    
    # Берём сбалансированную выборку по рейтингам
    df_balanced = pd.DataFrame()
    for rating in [1, 2, 3, 4, 5]:
        rating_df = df[df['Rating'] == rating]
        n_per_rating = min(n_samples // 5, len(rating_df))
        if n_per_rating > 0:
            df_balanced = pd.concat([df_balanced, rating_df.sample(n_per_rating, random_state=42)])
    
    # Если недостаточно, добираем случайными
    if len(df_balanced) < n_samples:
        additional = n_samples - len(df_balanced)
        remaining = df[~df.index.isin(df_balanced.index)]
        df_balanced = pd.concat([df_balanced, remaining.sample(min(additional, len(remaining)), random_state=42)])
    
    print(f"✅ Сбалансированная выборка: {len(df_balanced)} отзывов")
    
    for idx, row in df_balanced.iterrows():
        review_text = row['Review']
        rating = row['Rating']
        
        # Очищаем текст
        review_text = clean_text(review_text)
        if len(review_text) < 20:
            continue
        
        review = {
            "id": f"spotify_review_{idx}",
            "app": "Spotify",
            "rating": rating,
            "sentiment": map_rating_to_sentiment(rating),
            "text": review_text,
            "username": "Anonymous",
            "thumbsup": int(row['Total_thumbsup']) if 'Total_thumbsup' in row else 0,
            "source": "Google Play"
        }
        
        reviews.append(review)
    
    # Обрезаем до нужного количества
    reviews = reviews[:n_samples]
    print(f"✅ Подготовлено {len(reviews)} отзывов")
    return reviews

def save_reviews(reviews: list[dict], out_dir: str = "input"):
    """Сохранить отзывы в отдельные файлы"""
    
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    # Очищаем папку
    for f in out_path.glob("*.txt"):
        f.unlink()
    
    for review in reviews:
        content = f"""ID: {review['id']}
Приложение: {review['app']}
Пользователь: {review['username']}
Оценка: {review['rating']}/5 ({review['sentiment']})
Полезных: {review['thumbsup']}

Текст отзыва:
{review['text']}
"""
        file_path = out_path / f"{review['id']}.txt"
        file_path.write_text(content, encoding="utf-8")
    
    print(f"✅ Сохранено {len(reviews)} файлов в {out_dir}/")

def create_metadata(reviews: list[dict], out_dir: str = "input"):
    """Создать метаданные"""
    
    metadata = []
    for review in reviews:
        metadata.append({
            "id": review["id"],
            "app": review["app"],
            "rating": review["rating"],
            "sentiment": review["sentiment"],
            "text_length": len(review["text"]),
            "thumbsup": review["thumbsup"],
            "file": f"{review['id']}.txt"
        })
    
    metadata_path = Path(out_dir) / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ Метаданные сохранены: {metadata_path}")

def main():
    """Основной процесс"""
    
    print("━━━ Подготовка датасета Spotify отзывов ━━━\n")
    
    # Загружаем данные
    df = load_spotify_data("reviews.csv")
    
    # Показываем статистику
    show_stats(df)
    
    # Выбор количества отзывов
    print("\n" + "="*50)
    print("Варианты количества отзывов:")
    print("  1. 30 отзывов (очень быстро, ~$0.01)")
    print("  2. 50 отзывов (рекомендуется, ~$0.02)")
    print("  3. 100 отзывов (хорошо, ~$0.04)")
    print("  4. 200 отзывов (детально, ~$0.08)")
    print("  5. Своё число")
    print("="*50)
    
    choice = input("Выберите вариант (1-5): ").strip()
    
    if choice == "1":
        n_samples = 30
    elif choice == "2":
        n_samples = 50
    elif choice == "3":
        n_samples = 100
    elif choice == "4":
        n_samples = 200
    elif choice == "5":
        n_samples = int(input("Введите число: ").strip())
    else:
        n_samples = 50
    
    # Подготавливаем отзывы
    reviews = prepare_reviews(df, n_samples=n_samples)
    
    # Сохраняем
    save_reviews(reviews, "input")
    create_metadata(reviews, "input")
    
    # Оценка стоимости
    estimated_tokens = len(reviews) * 600  # Spotify отзывы короче
    estimated_cost = (estimated_tokens * 0.14 + estimated_tokens * 0.55) / 1_000_000
    print(f"\n💰 Ориентировочная стоимость: ${estimated_cost:.4f} (~{estimated_cost*100:.2f} центов)")
    
    print(f"\n━━━ Готово! ━━━")
    print(f"Теперь запусти: python pipeline.py input output")

if __name__ == "__main__":
    main()