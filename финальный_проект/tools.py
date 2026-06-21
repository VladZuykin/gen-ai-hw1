"""
Инструменты для поиска фильмов:
1. search_rag — поиск в RAG
2. search_web — поиск в интернете (DuckDuckGo)
3. get_full_page_text — извлечение полного текста со страницы
"""

import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from rag import MovieRAG
from config import DATA_DIR


# Глобальный экземпляр RAG (ленивая загрузка)
_RAG_INSTANCE: Optional[MovieRAG] = None

# Кэш для полного текста страниц (чтобы не ходить дважды)
_PAGE_CACHE: Dict[str, str] = {}


def clean_text(text: str) -> str:
    """Восстанавливает пробелы между словами после удаления HTML-тегов"""
    text = re.sub(r'([а-яА-Яa-zA-Z])([А-ЯA-Z])', r'\1 \2', text)
    text = re.sub(r'([а-яА-Яa-zA-Z])([а-яa-z])', r'\1 \2', text)
    return text


def get_full_page_text(url: str, max_chars: int = 5000, timeout: int = 10) -> Optional[str]:
    """
    Извлечь полный текст со страницы.
    
    Args:
        url: URL страницы
        max_chars: максимальное количество символов (чтобы не перегружать контекст)
        timeout: таймаут запроса
    
    Returns:
        Очищенный текст страницы или None при ошибке
    """
    # Проверяем кэш
    if url in _PAGE_CACHE:
        return _PAGE_CACHE[url]
    
    # Пропускаем не-html ссылки
    parsed = urlparse(url)
    if parsed.scheme not in ['http', 'https']:
        return None
    
    # Пропускаем видео/картинки
    if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi']):
        return None
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Добавляем небольшую задержку, чтобы не банили
        time.sleep(0.5)
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Парсим HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Удаляем скрипты, стили, навигацию
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # Удаляем лишние элементы
        for element in soup.find_all(["button", "form", "input"]):
            element.decompose()
        
        # Получаем текст
        text = soup.get_text(separator=' ', strip=True)
        
        # Очищаем от лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Обрезаем до max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + '...'
        
        # Сохраняем в кэш
        _PAGE_CACHE[url] = text
        
        return text
        
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке {url}: {e}")
        return None


def get_rag() -> MovieRAG:
    """Ленивая загрузка RAG"""
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = MovieRAG()
        
        # Пробуем загрузить из кэша
        cache_path = DATA_DIR / "rag_cache.json"
        if cache_path.exists():
            try:
                if _RAG_INSTANCE.load_cache(cache_path):
                    return _RAG_INSTANCE
            except Exception as e:
                print(f"⚠️ Ошибка загрузки кэша: {e}, переиндексируем...")
        
        _RAG_INSTANCE.load_from_csv(auto_download=True)
        _RAG_INSTANCE.save_cache(cache_path)
    
    return _RAG_INSTANCE


def search_rag(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Поиск фильмов в RAG"""
    rag = get_rag()
    results = rag.search(query, top_k=top_k)
    
    output = []
    for chunk, score in results:
        meta = chunk['metadata']
        output.append({
            'title': meta.get('title', 'Unknown'),
            'year': meta.get('year', 0),
            'plot': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text'],
            'similarity_score': float(score),
            'chunk_id': meta.get('chunk_id', '')
        })
    
    return output


def search_web(query: str, num_results: int = 3, get_full_text: bool = False) -> List[Dict[str, Any]]:
    """
    Поиск в интернете через DuckDuckGo.
    
    Args:
        query: Поисковый запрос
        num_results: Количество результатов
        get_full_text: Если True, загружает полный текст со страниц
    
    Returns:
        Список результатов с заголовками, URL, сниппетами и (опционально) полным текстом
    """
    if not query or not query.strip():
        return []
    
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.post(url, data=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result in soup.select('.result')[:num_results]:
            title_elem = result.select_one('.result__a')
            snippet_elem = result.select_one('.result__snippet')
            
            if title_elem:
                title = clean_text(title_elem.get_text(strip=True))
                link = title_elem.get('href', '')
                snippet = clean_text(snippet_elem.get_text(strip=True)) if snippet_elem else ""
                
                result_data = {
                    'title': title,
                    'url': link,
                    'snippet': snippet,
                    'source': 'web'
                }
                
                # Если нужен полный текст — загружаем
                if get_full_text and link:
                    full_text = get_full_page_text(link, max_chars=3000)
                    if full_text:
                        result_data['full_text'] = full_text
                
                results.append(result_data)
        
        return results
        
    except requests.exceptions.ConnectionError as e:
        print(f"⚠️ Ошибка соединения при веб-поиске: {e}")
        return []
    except requests.exceptions.Timeout as e:
        print(f"⚠️ Таймаут веб-поиска: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Ошибка веб-поиска: {e}")
        return []