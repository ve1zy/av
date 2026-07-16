# Описание решения

## Итоговая метрика

**MAP@10 (5-Fold Cross-Validation на calibration): 0.6117**

## Подход

**Learning to Rank (LTR)** с LightGBM LambdaRank на фичах от множества ранкеров + fine-tuned MiniLM.

### Предобработка

1. **Очистка HTML**: BeautifulSoup удаляет теги, скрипты и стили
2. **Усиление заголовка**: title повторяется 3 раза для увеличения веса
3. **Токенизация**: приведение к нижнему регистру, удаление спецсимволов
4. **Стемминг**: русский SnowballStemmer для BM25
5. **Стоп-слова**: удалены для одного из вариантов BM25

### Источники сигналов (ранкеры)

1. **BM25Okapi** — базовый keyword search (k1=2.0, b=0.5)
2. **BM25Okapi (stemmed)** — keyword search с русским стеммингом
3. **BM25Okapi (stopwords)** — keyword search без стоп-слов
4. **BM25Okapi (title)** — keyword search по заголовку
5. **TF-IDF** — с биграммами по полному тексту
6. **TF-IDF (title)** — с биграммами по заголовку
7. **Sentence embeddings (MiniLM, full)** — base + fine-tuned
8. **Sentence embeddings (MiniLM, title)** — base + fine-tuned
9. **Sentence embeddings (rubert-tiny2, full)**
10. **Sentence embeddings (rubert-tiny2, title)**
11. **Overlap features** — word/bigram/trigram Jaccard, token overlap

### Fine-tuning

MiniLM дообучается на hard triplets (query, positive, negative) из calibration с TripletLoss, 1 epoch.

### Модель ранжирования

LightGBM LambdaRank с регуляризацией:
- `num_leaves`: 31
- `learning_rate`: 0.02
- `num_boost_round`: 150
- `min_data_in_leaf`: 20
- `lambda_l1`: 1.0, `lambda_l2`: 1.0
- `feature_fraction`: 0.6
- `bagging_fraction`: 0.6

### Embedding модели

- `paraphrase-multilingual-MiniLM-L12-v2` (~22M параметров)
- `cointegrated/rubert-tiny2` (~29M параметров)

Обе модели локальные, open-source, значительно меньше 1B параметров.

## Используемые библиотеки

- `pandas` — работа с Feather/CSV
- `beautifulsoup4` — очистка HTML
- `rank_bm25` — BM25
- `scikit-learn` — TF-IDF
- `sentence-transformers` — эмбеддинги + fine-tuning
- `nltk` — стемминг и стоп-слова
- `lightgbm` — модель ранжирования
- `torch` — fine-tuning

## Воспроизведение

```bash
python solution_final_v4.py
```

Скрипт:
1. Загружает данные
2. Строит индексы BM25/TF-IDF
3. Кодирует статьи эмбеддингами (с кэшированием)
4. Fine-tune MiniLM на hard triplets
5. Собирает обучающую выборку из calibration
6. Обучает LightGBM LambdaRank
7. Генерирует `answer.csv`

## Что пробовалось

- Простой RRF: MAP@10 = 0.3861
- Бинарный LightGBM: MAP@10 = 0.5787
- LambdaRank: MAP@10 = 0.5999
- Grid search + LambdaRank: MAP@10 = 0.6117
- Fine-tuning MiniLM: улучшение на тесте
- Регуляризация LGBM: уменьшение overfit

## Ограничения

- Все модели запускаются локально
- Без внешних API и LLM >1B параметров
- Без ручной разметки тестовых данных
- Calibration используется только для обучения/валидации, не test
