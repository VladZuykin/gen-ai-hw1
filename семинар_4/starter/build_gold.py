"""
Расширенный скрипт для проверки чанков и составления gold.json.
Показывает полный текст каждого чанка для осознанного выбора.
"""

import json
from pathlib import Path

def load_chunks():
    with open("bm25_cache.json", "r", encoding="utf-8") as f:
        return json.load(f)

def show_full_chunk(data, chunk_id):
    """Показывает ПОЛНЫЙ текст чанка"""
    for idx, cid in enumerate(data["ids"]):
        if cid == chunk_id:
            print(f"\n{'='*70}")
            print(f"📄 ПОЛНЫЙ ТЕКСТ ЧАНКА: {cid}")
            print(f"{'='*70}")
            print(data["texts"][idx])
            print(f"{'='*70}\n")
            return True
    print(f"❌ Чанк {chunk_id} не найден")
    return False

def search_and_select(data, question):
    """Ищет чанки по ключевым словам и позволяет выбрать"""
    print(f"\n{'='*70}")
    print(f"❓ [{question['id']}] {question['type']}: {question['question']}")
    print(f"{'='*70}")
    
    # Поиск чанков
    results = []
    for idx, (cid, text) in enumerate(zip(data["ids"], data["texts"])):
        text_lower = text.lower()
        matched = [kw for kw in question['keywords'] if kw.lower() in text_lower]
        if matched:
            results.append({
                "id": cid,
                "matched": matched,
                "preview": text[:300].replace("\n", " ") + "..."
            })
    
    if not results:
        print("\n⚠️ НИЧЕГО НЕ НАЙДЕНО по ключевым словам")
        print("   Это может быть ловушкой или нужно добавить ключевые слова")
        return []
    
    print(f"\n📊 Найдено {len(results)} чанков:")
    for i, r in enumerate(results):
        print(f"\n   [{i+1}] {r['id']}")
        print(f"       Ключевые слова: {r['matched']}")
        print(f"       Превью: {r['preview']}")
    
    # Показываем полные тексты
    print(f"\n{'='*70}")
    print("🔍 ПРОВЕРКА ПОЛНЫХ ТЕКСТОВ")
    print(f"{'='*70}")
    
    selected = []
    for r in results[:3]:  # показываем топ-3
        print(f"\n--- Чанк: {r['id']} ---")
        show_full_chunk(data, r['id'])
        
        answer = input(f"Добавить {r['id']} в gold_sources? (y/n): ").strip().lower()
        if answer == 'y':
            selected.append(r['id'])
    
    # Если нужно посмотреть ещё чанки
    if len(results) > 3:
        more = input(f"\nОсталось {len(results)-3} чанков. Посмотреть ещё? (y/n): ").strip().lower()
        if more == 'y':
            for r in results[3:]:
                print(f"\n--- Чанк: {r['id']} ---")
                show_full_chunk(data, r['id'])
                answer = input(f"Добавить в gold_sources? (y/n): ").strip().lower()
                if answer == 'y':
                    selected.append(r['id'])
    
    return selected

def main():
    print("Загрузка чанков из bm25_cache.json...")
    data = load_chunks()
    print(f"✅ Загружено {len(data['ids'])} чанков\n")
    
    questions = [
        {
            "id": 1,
            "type": "прямой",
            "question": "Кто сформулировал теорему о сжимающих отображениях?",
            "keywords": ["банах", "теорема банаха", "сжима", "неподвижн"]
        },
        {
            "id": 2,
            "type": "прямой",
            "question": "Что такое метод касательных Ньютона?",
            "keywords": ["ньютон", "касательн", "xn+1", "x_{n+1}", "метод ньютона"]
        },
        {
            "id": 3,
            "type": "синоним",
            "question": "Какая теорема гарантирует существование неподвижной точки у сжимающего отображения?",
            "keywords": ["банах", "неподвижн", "сжима", "теорем"]
        },
        {
            "id": 4,
            "type": "точный артикул",
            "question": "Что такое ФЗ-152?",
            "keywords": ["фз-152", "115-фз", "федеральн", "закон"]
        },
        {
            "id": 5,
            "type": "multi-hop",
            "question": "Какие два метода решения уравнений с одним неизвестным обсуждались?",
            "keywords": ["метод деления", "метод касательных", "половин", "ньютон", "бессера"]
        },
        {
            "id": 6,
            "type": "позитив",
            "question": "Какие математические результаты были названы важными или замечательными?",
            "keywords": ["важн", "замечательн", "значим", "велик"]
        },
        {
            "id": 7,
            "type": "перефраз",
            "question": "У кого производная не обращается в ноль в точке экстремума?",
            "keywords": ["производн", "экстремум", "ноль", "не обращ"]
        },
        {
            "id": 8,
            "type": "прямой",
            "question": "Что такое равномерная сходимость функционального ряда?",
            "keywords": ["равномерн", "сходимост", "функциональн", "ряд"]
        },
        {
            "id": 9,
            "type": "синоним",
            "question": "Как называется формула, связывающая экспоненту с синусом и косинусом?",
            "keywords": ["эйлер", "формула эйлера", "e^", "синус", "косинус"]
        },
        {
            "id": 10,
            "type": "multi-hop",
            "question": "Какие две теоремы используются для доказательства существования обратного отображения?",
            "keywords": ["обратн", "отображен", "теорема об обратном", "банах", "сжима"]
        },
        {
            "id": 11,
            "type": "точный артикул",
            "question": "Что такое формула Эйлера?",
            "keywords": ["эйлер", "e^ix", "cos", "sin", "формула эйлера"]
        },
        {
            "id": 12,
            "type": "перефраз",
            "question": "Какой метод позволяет находить приближённое решение уравнения, если функция принимает разные знаки на концах отрезка?",
            "keywords": ["делен", "пополам", "бальзан", "коши", "разные знаки"]
        },
        {
            "id": 13,
            "type": "multi-hop",
            "question": "Какие два условия необходимы для применения метода Ньютона к решению уравнения?",
            "keywords": ["ньютон", "услови", "производн", "сходимост", "c^2", "гладк"]
        },
        {
            "id": 14,
            "type": "multi-hop",
            "question": "В чём разница между точечной и равномерной сходимостью последовательности функций?",
            "keywords": ["точечн", "равномерн", "сходимост", "разниц", "супремум"]
        },
        {
            "id": 15,
            "type": "multi-hop",
            "question": "Какие свойства сохраняет сумма ряда при равномерной сходимости?",
            "keywords": ["равномерн", "свойств", "непрерывн", "дифференц", "наследует"]
        },
        {
            "id": 16,
            "type": "multi-hop",
            "question": "Почему экспонента не является взаимно однозначным отображением, и как эту проблему решают?",
            "keywords": ["экспонент", "взаимн", "однозначн", "период", "логарифм", "многозначн"]
        },
        {
            "id": 17,
            "type": "multi-hop",
            "question": "Какая связь между теоремой Банаха и методом итераций для решения уравнений?",
            "keywords": ["банах", "итерац", "неподвижн", "сжима", "последовательн"]
        }
    ]
    
    gold_entries = []
    
    for q in questions:
        selected = search_and_select(data, q)
        
        gold_entries.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "gold_sources": selected
        })
        
        print(f"\n✅ Для вопроса {q['id']} выбрано: {selected if selected else 'НИЧЕГО (ловушка)'}")
        input("\nНажмите Enter для продолжения...")
    
    # Сохраняем результат
    output_path = Path("data/gold.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_entries, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ gold.json сохранён в {output_path}")
    print(f"   Всего вопросов: {len(gold_entries)}")
    print(f"{'='*70}")
    
    # Статистика
    print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    for entry in gold_entries:
        status = "✓" if entry["gold_sources"] else "✗ (ловушка)"
        print(f"  [{entry['id']:2d}] {entry['type']:20s} {status} -> {entry['gold_sources']}")

if __name__ == "__main__":
    main()