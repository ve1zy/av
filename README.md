# Avito Support Article Search

Решение задачи поиска релевантных статей справочного центра Авито по пользовательским запросам.

## Метрика

**MAP@10 на calibration set (5-Fold CV): 0.6117**

## Подход

**Learning to Rank (LTR)** с LightGBM LambdaRank + fine-tuned MiniLM на фичах от множества ранкеров.

### Ранкеры (источники фич)

- BM25 базовый (k1=2.0, b=0.5)
- BM25 с русским стеммингом
- BM25 без стоп-слов
- BM25 по заголовку
- TF-IDF с биграммами (full + title)
- Sentence embeddings MiniLM (full + title) — base + fine-tuned
- Sentence embeddings rubert-tiny2 (full + title)
- Overlap features: word/bigram/trigram Jaccard, token overlap counts

### Модель ранжирования

LightGBM LambdaRank с регуляризацией против overfit:
- `num_leaves`: 31
- `learning_rate`: 0.02
- `num_boost_round`: 150
- `min_data_in_leaf`: 20
- `lambda_l1`: 1.0, `lambda_l2`: 1.0

### Fine-tuning

MiniLM дообучается на hard triplets из calibration (1 epoch, TripletLoss).

### Embedding модели

- `paraphrase-multilingual-MiniLM-L12-v2` (~22M параметров)
- `cointegrated/rubert-tiny2` (~29M параметров)

Обе модели локальные, open-source, <1B параметров.

## Файлы

- `solution_final_v4.py` — финальный скрипт
- `answer.csv` — ответы для test.f
- `requirements.txt` — зависимости
- `approach.md` — подробное описание решения

## Запуск

```bash
pip install -r requirements.txt
python solution_final_v4.py
```

## Структура данных

В папке `candidate_data/` должны быть:
- `articles.f`
- `calibration.f`
- `test.f`

## Кэширование

Embeddings кэшируются в `cache/` для ускорения повторных запусков.
