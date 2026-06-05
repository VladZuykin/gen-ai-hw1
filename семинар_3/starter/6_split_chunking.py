"""
Раунд 3a — Стратегии разбиения
==============================
Map-Reduce начинается с разбиения. У нашего транскрипта 4 говорящих и
~150 строк — для одного запроса разбиение вообще не нужно. Но как только
он вырастет до 200k токенов — придётся резать. И тут возникает вопрос:
КАК резать?

Задача:
  Реализовать ТРИ функции разбиения. На нашем коротком transcript.txt
  посчитать: сколько фрагментов, средний размер, минимум/максимум.
  Это даст почувствовать разницу подходов «руками».

Запуск:
    python 6_split_chunking.py
"""

from __future__ import annotations

import re
from pathlib import Path

SPEAKER_RE = re.compile(r"^([А-ЯЁA-Z][а-яёa-zА-ЯЁA-Z]+):\s", re.MULTILINE)
_SKIP_SPEAKERS = {"Модератор", "Moderator", "Дата", "Участники", "Date"}


def split_by_speaker(transcript: str) -> list[str]:
    """Разбить по говорящему. Шапку (поля Дата/Участники/...) выкинуть.

    Подсказка: текст до первого "═══" — это метаданные, режь их.
    """
    lines = transcript.split('\n')
    
    # Находим начало после метаданных
    start_idx = 0
    for i, line in enumerate(lines):
        if '═══' in line:
            start_idx = i + 1
            break
    
    # Собираем всё, что говорит каждый участник
    speaker_chunks = {}
    current_speaker = None
    current_text = []
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        match = SPEAKER_RE.match(line)
        if match:
            # Сохраняем предыдущего speaker
            if current_speaker and current_text:
                if current_speaker not in speaker_chunks:
                    speaker_chunks[current_speaker] = []
                speaker_chunks[current_speaker].extend(current_text)
            
            # Начинаем нового speaker
            speaker = match.group(1)
            if speaker in _SKIP_SPEAKERS:
                current_speaker = None
                current_text = []
            else:
                current_speaker = speaker
                current_text = [line]
        elif current_speaker:
            current_text.append(line)
    
    # Добавляем последнего speaker
    if current_speaker and current_text:
        if current_speaker not in speaker_chunks:
            speaker_chunks[current_speaker] = []
        speaker_chunks[current_speaker].extend(current_text)
    
    # Формируем чанки: один чанк = все реплики одного участника
    chunks = []
    for speaker, lines_list in speaker_chunks.items():
        chunk = '\n'.join(lines_list)
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def split_by_chars(transcript: str, max_chars: int = 1500) -> list[str]:
    """Разбить на куски по N символов, стараясь резать по \\n\\n."""
    # Убираем метаданные
    lines = transcript.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if '═══' in line:
            start_idx = i + 1
            break
    
    content = '\n'.join(lines[start_idx:])
    paragraphs = content.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        # Если один параграф больше max_chars, режем принудительно
        if para_size > max_chars:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Режем большой параграф по предложениям
            sentences = para.replace('!', '.').replace('?', '.').split('.')
            temp_chunk = []
            temp_size = 0
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent += '.'
                if temp_size + len(sent) > max_chars and temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = [sent]
                    temp_size = len(sent)
                else:
                    temp_chunk.append(sent)
                    temp_size += len(sent)
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
        
        # Если новый параграф влезает в текущий чанк
        elif current_size + para_size + 2 <= max_chars:
            current_chunk.append(para)
            current_size += para_size + 2
        else:
            # Сохраняем текущий чанк и начинаем новый
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def split_sliding(transcript: str, window: int = 1500, overlap: int = 300) -> list[str]:
    """Скользящее окно: каждый следующий кусок начинается на (window-overlap)
    символов позже. Полезно, когда смысловые единицы могут пересекать
    границы фиксированного разбиения.
    """
    # Убираем метаданные
    lines = transcript.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if '═══' in line:
            start_idx = i + 1
            break
    
    content = '\n'.join(lines[start_idx:])
    step = window - overlap
    
    if step <= 0:
        step = window // 2  # fallback, если overlap >= window
    
    chunks = []
    for start in range(0, len(content), step):
        end = min(start + window, len(content))
        chunk = content[start:end]
        chunks.append(chunk)
        
        if end == len(content):
            break
    
    return chunks


def stats(chunks: list[str], label: str) -> None:
    if not chunks:
        print(f"  {label:<20} 0 фрагментов")
        return
    sizes = [len(c) for c in chunks]
    print(
        f"  {label:<20} n={len(chunks):<3} "
        f"средн.={sum(sizes) // len(sizes):>5} "
        f"мин={min(sizes):>5} макс={max(sizes):>5}"
    )


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    print(f"Транскрипт: {len(transcript)} символов\n")

    print("━━━ Три стратегии разбиения ━━━")
    stats(split_by_speaker(transcript), "по говорящим")
    stats(split_by_chars(transcript, max_chars=1500), "по размеру (1500)")
    stats(split_sliding(transcript, window=1500, overlap=300), "скользящее (1500,300)")

    print("\nЧто обсудить:")
    print(
        "  • для нашего короткого транскрипта разбиение по говорящим — очевидный выбор;"
    )
    print(
        "  • для 100-страничного отчёта разбиение по размеру даст однородные фрагменты;"
    )
    print("  • для длинного монолога эксперта скользящее окно не потеряет контекст.")


if __name__ == "__main__":
    main()
