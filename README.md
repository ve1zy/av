# Avito Support Article Search

Решение задачи поиска релевантных статей справочного центра Авито по пользовательским запросам.

## Метрика

**MAP@10 на calibration set (5-Fold CV): 0.6117**

## Подход

**Learning to Rank (LTR)** с LightGBM LambdaRank на фичах от множества ранкеров.

### Ранкеры (источники фич)

- BM25 базовый
- BM25 с русским стеммингом
- BM25 без стоп-слов
- BM25 по заголовку
- TF-IDF с биграммами
- TF-IDF по заголовку
- Sentence embeddings (MiniLM) по полному тексту
- Sentence embeddings (MiniLM) по заголовку
- Sentence embeddings (rubert-tiny2) по полному тексту
- Sentence embeddings (rubert-tiny2) по заголовку

### Модель ранжирования

LightGBM LambdaRank обучается на 100 hard negatives на запрос из топа кандидатов.

Лучшие параметры:
- `num_leaves`: 127
- `learning_rate`: 0.03
- `num_boost_round`: 300

Для каждой пары формируются фичи:
- Сырые scores от каждого ранкера
- Нормализованные scores
- Ранги
- Длины запроса и документа
- Interaction features между ранкерами

### Embedding модели

- `paraphrase-multilingual-MiniLM-L12-v2` (~22M параметров)
- `cointegrated/rubert-tiny2` (~29M параметров)

Обе модели локальные, open-source, <1B параметров.

## Файлы

- `solution_v16_final.py` — финальный скрипт (генерация answer.csv)
- `solution_v16.py` — grid search по LightGBM параметрам
- `solution_v14.py` — LambdaRank с CV
- `answer.csv` — ответы для test.f
- `requirements.txt` — зависимости
- `approach.md` — подробное описание решения

## Запуск

```bash
pip install -r requirements.txt
python solution_v16_final.py
```

## Структура данных

В папке `candidate_data/` должны быть:
- `articles.f`
- `calibration.f`
- `test.f`

## Воспроизводимость

Скрипт `solution_v16_final.py` использует фиксированные параметры BM25 и LightGBM. LightGBM с bagging может давать небольшие вариации, но основной сигнал стабилен.
