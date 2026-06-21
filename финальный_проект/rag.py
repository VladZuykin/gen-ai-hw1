"""
RAG с гибридным поиском (BM25 + Dense + RRF).
Адаптировано из семинара 4 для поиска фильмов.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


class MovieRAG:
    """RAG для поиска фильмов по описанию сюжета"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.bm25 = None
        self.chunks = []
        self.embeddings = None
        self._is_loaded = False
        
    def load_from_csv(
        self, 
        csv_path: Optional[Path] = None,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        max_films: Optional[int] = None,
        use_summary: bool = True,
    ):
        """
        Загрузить датасет фильмов и подготовить чанки.
        
        Args:
            csv_path: путь к CSV
            chunk_size: размер чанка
            overlap: перекрытие
            max_films: максимальное количество фильмов (None = все)
            use_summary: добавлять summary к plot
        """
        if self._is_loaded:
            print("⚠️ RAG уже загружен, пропускаем")
            return
        
        if csv_path is None:
            csv_path = DATA_DIR / "movies.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Датасет не найден: {csv_path}")
        
        # Загружаем данные
        df = pd.read_csv(csv_path)
        total_films = len(df)
        print(f"Загружено {total_films} фильмов")
        
        # Ограничиваем количество
        if max_films is not None:
            df = df.head(max_films)
            print(f"   Взято {len(df)} фильмов (первые {max_films})")
        
        # Определяем колонки
        title_col = None
        plot_col = None
        summary_col = None
        genre_col = None
        year_col = None
        
        for col in ['title', 'Title', 'name', 'Name', 'film', 'Film']:
            if col in df.columns:
                title_col = col
                break
        
        for col in ['plot', 'Plot']:
            if col in df.columns:
                plot_col = col
                break
        
        for col in ['summary', 'Summary', 'description', 'Description']:
            if col in df.columns:
                summary_col = col
                break
        
        for col in ['genre', 'Genre', 'genres', 'Genres']:
            if col in df.columns:
                genre_col = col
                break
        
        for col in ['year', 'Year', 'release_year', 'Release_Year']:
            if col in df.columns:
                year_col = col
                break
        
        if plot_col is None:
            raise ValueError(f"Колонка с сюжетом не найдена. Доступны: {list(df.columns)}")
        
        if title_col is None:
            title_col = plot_col
        
        print(f"   Использую колонки:")
        print(f"      title: '{title_col}'")
        print(f"      plot: '{plot_col}'")
        if summary_col:
            print(f"      summary: '{summary_col}' (будет объединён с plot)")
        if genre_col:
            print(f"      genre: '{genre_col}'")
        if year_col:
            print(f"      year: '{year_col}'")
        
        # Создаём чанки из сюжетов
        self.chunks = []
        skipped = 0
        
        total = len(df)
        print(f"\n⏳ Обработка фильмов...")
        
        for idx, row in df.iterrows():
            # Показываем прогресс каждые 100 фильмов
            if (idx + 1) % 100 == 0 or idx == 0 or idx == total - 1:
                print(f"   [{idx + 1}/{total}] ({((idx + 1)/total*100):.1f}%)", end='\r')
            
            plot = row.get(plot_col, '')
            if pd.isna(plot) or not str(plot).strip():
                skipped += 1
                continue
            
            # Собираем полный текст для индексации
            full_text = str(plot)
            
            if use_summary and summary_col:
                summary = row.get(summary_col, '')
                if not pd.isna(summary) and str(summary).strip():
                    full_text = f"{summary} {full_text}"
            
            if genre_col:
                genre = row.get(genre_col, '')
                if not pd.isna(genre) and str(genre).strip():
                    full_text = f"Жанр: {genre}. {full_text}"
            
            title = str(row.get(title_col, 'Unknown'))
            
            year = 0
            if year_col and not pd.isna(row.get(year_col)):
                try:
                    year = int(row.get(year_col))
                except (ValueError, TypeError):
                    pass
            
            rating = 0.0
            if 'imdb_rating' in df.columns and not pd.isna(row.get('imdb_rating')):
                try:
                    rating = float(row.get('imdb_rating'))
                except (ValueError, TypeError):
                    pass
            
            chunks = self._recursive_chunking(full_text, chunk_size, overlap)
            for i, chunk_text in enumerate(chunks):
                self.chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'title': title,
                        'year': year,
                        'genre': str(genre) if genre_col else '',
                        'rating': rating,
                        'chunk_id': f"{title.replace(' ', '_')}__{i}"
                    }
                })
        
        print(f"\n✅ Обработано {total} фильмов")
        print(f"   Создано {len(self.chunks)} чанков (пропущено {skipped} без сюжета)")
        
        # Индексация
        print("\n⏳ Индексация BM25...")
        self._index_bm25()
        print("⏳ Индексация Dense (эмбеддинги)...")
        self._index_dense()
        self._is_loaded = True
        print("✅ Индексация завершена")
        
    def _recursive_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Recursive chunking с перекрытием"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            sent_len = len(sent.split())
            if current_size + sent_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_sentences = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_sentences
                current_size = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sent)
            current_size += sent_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _index_bm25(self):
        """BM25 индексация"""
        print(f"   Токенизация {len(self.chunks)} чанков...")
        tokenized_chunks = []
        for i, chunk in enumerate(self.chunks):
            if (i + 1) % 5000 == 0 or i == 0 or i == len(self.chunks) - 1:
                print(f"      Токенизация: {i+1}/{len(self.chunks)}", end='\r')
            tokenized_chunks.append(self._tokenize(chunk['text']))
        print(f"      Токенизация: {len(self.chunks)}/{len(self.chunks)} ✅")
        
        print("   Построение BM25 индекса...")
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("   BM25 индекс готов ✅")
    
    def _index_dense(self):
        """Dense индексация (эмбеддинги)"""
        texts = [chunk['text'] for chunk in self.chunks]
        print(f"   Генерация эмбеддингов для {len(texts)} текстов...")
        self.embeddings = self.model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print("   Эмбеддинги готовы ✅")
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация для BM25"""
        tokens = re.findall(r'\w+', text.lower())
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Гибридный поиск с RRF"""
        if not self._is_loaded:
            raise RuntimeError("RAG не загружен. Сначала вызови load_from_csv()")
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.embeddings, query_embedding)
        
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        dense_rank = np.argsort(-dense_scores)
        bm25_rank = np.argsort(-bm25_scores)
        
        rrf_scores = {}
        k = 60
        
        for rank, idx in enumerate(dense_rank[:top_k*2]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        
        for rank, idx in enumerate(bm25_rank[:top_k*2]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        
        sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = [(self.chunks[idx], rrf_scores[idx]) for idx in sorted_idx[:top_k]]
        
        return results
    
    def save_cache(self, path: Path):
        """Сохранить кэш"""
        if not self._is_loaded:
            return
        
        print(f"\n💾 Сохранение кэша в {path}...")
        embeddings_list = None
        if self.embeddings is not None:
            embeddings_list = self.embeddings.tolist()
        
        data = {
            'chunks': self.chunks,
            'embeddings': embeddings_list,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Кэш сохранён: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    def load_cache(self, path: Path):
        """Загрузить кэш"""
        if not path.exists():
            return False
        
        print(f"\n📂 Загрузка кэша из {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        if data.get('embeddings'):
            self.embeddings = np.array(data['embeddings'])
        
        print("   Восстановление BM25 индекса...")
        self._index_bm25()
        self._is_loaded = True
        print(f"✅ Кэш загружен: {len(self.chunks)} чанков")
        return True