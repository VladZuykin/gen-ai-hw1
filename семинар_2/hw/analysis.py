"""
Финал семинара — расширенный анализ качества данных
====================================================
Анализ заявок на курсы повышения квалификации.

Что считаем:
  1. Гистограмма возрастов — была и раньше.
  2. Распределение по городам и специальностям (бары) — найдём mode collapse.
  3. Топ-N повторяющихся имён — другая грань collapse (модель любит «Анну»).
  4. Кросс-таблица город × специальность — есть ли нереалистичные комбинации?
  5. Boxplot опыт × специальность — модель умеет связывать поля или просто генерит независимо?

На выходе:
  - ages.png         — гистограмма возрастов
  - cities.png       — распределение по городам
  - specialities.png — распределение по специальностям
  - experience_by_speciality.png — boxplot
  - report.md        — текстовая сводка для обсуждения

Запуск:
  python analysis.py applications.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        sys.exit("Файл пустой — сначала прогоните generator.py")
    flat = []
    for item in data:
        row = dict(item)
        if isinstance(row.get("address"), dict):
            addr = row.pop("address")
            row["city"] = addr.get("city")
            row["district"] = addr.get("district")
        flat.append(row)
    return pd.DataFrame(flat)


def plot_hist_ages(df: pd.DataFrame, out: str):
    plt.figure(figsize=(8, 4))
    plt.hist(df["age"], bins=12, color="#4A90D9", edgecolor="white")
    plt.xlabel("Возраст")
    plt.ylabel("Число заявок")
    plt.title(f"Распределение возраста ({len(df)} заявок)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_bar(series: pd.Series, title: str, out: str, color="#4A90D9"):
    counts = series.value_counts()
    plt.figure(figsize=(9, 4))
    counts.plot.bar(color=color, edgecolor="white")
    plt.title(title)
    plt.ylabel("Число заявок")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return counts


def plot_experience_by_speciality(df: pd.DataFrame, out: str):
    if "years_of_experience" not in df.columns or "speciality" not in df.columns:
        return
    groups = df.groupby("speciality")["years_of_experience"].apply(list)
    plt.figure(figsize=(10, 4))
    positions = range(1, len(groups) + 1)
    plt.boxplot(list(groups.values), positions=list(positions), vert=True)
    plt.xticks(list(positions), list(groups.index), rotation=30, ha="right")
    plt.ylabel("Опыт, лет")
    plt.title("Опыт × специальность")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def cross_table(df: pd.DataFrame, col_1, col_2) -> pd.DataFrame:
    if col_1 not in df.columns or col_2 not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df[col_1], df[col_2])


def write_report(df: pd.DataFrame, out: str):
    n = len(df)
    lines = [f"# Отчёт по {n} заявкам\n"]

    # Топ городов
    cities = df["city"].value_counts()
    top_city_pct = cities.iloc[0] / n * 100
    lines.append("## Города\n")
    lines.append(f"- Уникальных: {len(cities)}")
    lines.append(f"- Топ-1: **{cities.index[0]}** — {cities.iloc[0]} ({top_city_pct:.0f}%)")
    if top_city_pct > 40:
        lines.append(f"- ⚠ Превышен порог 40% → mode collapse по городам")
    lines.append("")

    # Топ специальностей
    spec = df["speciality"].value_counts()
    top_spec_pct = spec.iloc[0] / n * 100
    lines.append("## Специальности\n")
    lines.append(f"- Уникальных: {len(spec)}")
    lines.append(f"- Топ-1: **{spec.index[0]}** — {spec.iloc[0]} ({top_spec_pct:.0f}%)")
    if top_spec_pct > 35:
        lines.append(f"- ⚠ Превышен порог 35% → mode collapse по специальностям")
    lines.append("")

    # Дубликаты имён
    names = df["full_name"].value_counts()
    dupes = names[names > 1]
    lines.append("## Имена\n")
    lines.append(f"- Уникальных: {len(names)} из {n} ({len(names)/n*100:.0f}%)")
    if len(dupes):
        lines.append(f"- Повторы: {dict(dupes.head(5))}")
    else:
        lines.append("- Повторов нет")
    lines.append("")

    # Кросс-таблица
    ct = cross_table(df, 'age', 'years_of_experience')
    if not ct.empty:
        lines.append("## Кросс-таблица\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        for city in cities.head(2).index:
            row = ct.loc[city] if city in ct.index else None
            if row is not None:
                empty = row[row == 0].index.tolist()
                if empty:
                    lines.append(f"- В **{city}** ни одного: {', '.join(empty)}")
        lines.append("")


    # Кросс-таблица
    ct = cross_table(df, 'years_of_experience', 'graduation_year')
    if not ct.empty:
        lines.append("## Кросс-таблица\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        for city in cities.head(2).index:
            row = ct.loc[city] if city in ct.index else None
            if row is not None:
                empty = row[row == 0].index.tolist()
                if empty:
                    lines.append(f"- В **{city}** ни одного: {', '.join(empty)}")
        lines.append("")


    # Кросс-таблица
    ct = cross_table(df, 'desired_course', 'speciality')
    if not ct.empty:
        lines.append("## Кросс-таблица\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        for city in cities.head(2).index:
            row = ct.loc[city] if city in ct.index else None
            if row is not None:
                empty = row[row == 0].index.tolist()
                if empty:
                    lines.append(f"- В **{city}** ни одного: {', '.join(empty)}")
        lines.append("")


    # Кросс-таблица
    ct = cross_table(df, 'city', 'district')
    if not ct.empty:
        lines.append("## Кросс-таблица\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        for city in cities.head(2).index:
            row = ct.loc[city] if city in ct.index else None
            if row is not None:
                empty = row[row == 0].index.tolist()
                if empty:
                    lines.append(f"- В **{city}** ни одного: {', '.join(empty)}")
        lines.append("")

    # Опыт × специальность
    if "years_of_experience" in df.columns:
        med = df.groupby("speciality")["years_of_experience"].median().sort_values(ascending=False)
        lines.append("## Медианный опыт по специальностям\n")
        for spec_name, m in med.items():
            lines.append(f"- {spec_name}: {int(m):.0f} лет")
        lines.append("")

    Path(out).write_text("\n".join(lines), encoding="utf-8")


def main(path: str = "applications.json"):
    df = load(path)
    print(f"Загружено: {len(df)} заявок из {path}")

    plot_hist_ages(df, "ages.png")
    c = plot_bar(df["city"], "Распределение по городам", "cities.png", "#7AB66E")
    s = plot_bar(df["speciality"], "Распределение по специальностям", "specialities.png", "#D97A4A")
    plot_experience_by_speciality(df, "experience_by_speciality.png")
    write_report(df, "report.md")

    print("\nСохранено:")
    for f in ("ages.png", "cities.png", "specialities.png", "experience_by_speciality.png", "report.md"):
        if Path(f).exists():
            print(f"  - {f}")

    print(f"\nТоп-город: {c.index[0]} ({c.iloc[0]}/{len(df)})")
    print(f"Топ-специальность: {s.index[0]} ({s.iloc[0]}/{len(df)})")
    print("\nДальше — открыть report.md и обсудить с группой:")
    print("  - где collapse, какое поле «слиплось» сильнее всего?")
    print("  - есть ли нереалистичные комбинации в кросс-таблице?")
    print("  - модель связывает опыт с специальностью или генерит независимо?")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "applications.json"
    main(path)