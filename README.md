# Avito Support Article Search

Решение задачи поиска релевантных статей справочного центра Авито по пользовательским запросам.

## Метрика

**MAP@10 на calibration set (5-Fold CV): 0.5787**

## Подход

**Learning to Rank (LTR)** с LightGBM на фичах от множества ранкеров.

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

LightGBM обучается на парах (запрос, кандидат) с hard negatives из топ-50 кандидатов.

Для каждой пары формируются фичи:
- Сырые scores от каждого ранкера
- Нормализованные scores
- Ранги
- Длины запроса и документа

### Embedding модели

- `paraphrase-multilingual-MiniLM-L12-v2` (~22M параметров)
- `cointegrated/rubert-tiny2` (~29M параметров)

Обе модели локальные, open-source, <1B параметров.

## Файлы

- `solution_v13_final.py` — финальный скрипт (обучение + генерация ответа)
- `solution_v13_cv.py` — скрипт 5-fold кросс-валидации
- `answer.csv` — ответы для test.f
- `requirements.txt` — зависимости
- `approach.md` — подробное описание решения

## Запуск

```bash
pip install -r requirements.txt
python solution_v13_final.py
```

## Структура данных

В папке `candidate_data/` должны быть:
- `articles.f`
- `calibration.f`
- `test.f`

## Воспроизводимость

Скрипт `solution_v13_final.py` детерминирован: фиксированные параметры BM25, фиксированный seed отсутствует, но LightGBM с bagging может давать небольшие вариации. Основной сигнал стабилен.
