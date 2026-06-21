
## Команда запуска: скачать датасет + индексация 3000 фильмов + eval на 3-х фильмах

```bash
./start.sh
```

---

## 📁 Структура проекта

```
project/
├── start.sh             # ОДНА КОМАНДА ЗАПУСКА
├── README.md            # Этот файл
├── requirements.txt     # Зависимости
├── .env.example         # Шаблон .env (без токена)
│
├── llm_client.py        # Клиент + трекинг стоимости
├── schemas.py           # Pydantic + field_validator (год 1888-2025)
├── rag.py               # RAG (BM25 + Dense + RRF)
├── tools.py             # search_rag, search_web
├── planner.py           # Планировщик
├── critic.py            # LLM-as-Judge и critic в одном лице (проверка галлюцинаций)
├── orchestrator.py      # Мультиагент (главный цикл)
├── executor.py          # Исполнитель
├── eval.py              # 15 тестов
├── download_dataset.py  # Скачивание датасета с Kaggle
├── index_rag.py         # Индексация RAG
├── utils.py             # Логирование
├── config.py            # Конфигурация
│
├── input/
│   ├── movies.csv       # Датасет с download_dataset.py
│   └── rag_cache.json   # Кэш RAG
│
└── output/
    ├── eval_results.json # Результаты eval
    └── trace_*.json      # Трейсы каждого запроса
```

---

## 🎬 Команды

| Команда | Что делает |
|---------|------------|
| `python download_dataset.py` | Скачать датасет с Kaggle |
| `python index_rag.py --max 3000` | Индексировать 3 000 фильмов |
| `python orchestrator.py "описание"` | Найти фильм по описанию |
| `python eval.py` | Запустить все 15 тестов |
| `python eval.py --ids 1 2 3` | Запустить конкретные тесты |

---

## 📊 Пример

```bash
python orchestrator.py "Парень узнаёт, что живёт в матрице"
```