# Avito Support Article Search

Решение задачи поиска релевантных статей справочного центра Авито по пользовательским запросам.

## Метрика

**MAP@10 на calibration set: 0.3861**

## Подход

Гибридный поиск с шестью ранкерами:
- BM25 базовый
- BM25 с русским стеммингом
- BM25 без стоп-слов
- TF-IDF с биграммами
- Sentence embeddings по полному тексту
- Sentence embeddings по заголовку

Объединение через Reciprocal Rank Fusion (RRF).

Модель эмбеддингов: `paraphrase-multilingual-MiniLM-L12-v2` (локальная, ~22M параметров).

## Файлы

- `solution_final_fast.py` — быстрый финальный скрипт
- `solution_final.py` — версия с grid search параметров
- `answer.csv` — ответы для test.f
- `requirements.txt` — зависимости
- `approach.md` — подробное описание решения

## Запуск

```bash
pip install -r requirements.txt
python solution_final_fast.py
```

## Структура данных

В папке `candidate_data/` должны быть:
- `articles.f`
- `calibration.f`
- `test.f`
