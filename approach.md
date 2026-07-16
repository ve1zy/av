# Описание решения

## Итоговая метрика

**MAP@10 (5-Fold Cross-Validation на calibration): 0.6117**

Train MAP@10 (обучение на всей calibration): 0.9787

## Подход

**Learning to Rank (LTR)** с LightGBM LambdaRank на фичах от множества ранкеров.

### Предобработка

1. **Очистка HTML**: BeautifulSoup удаляет теги, скрипты и стили
2. **Усиление заголовка**: title повторяется 3 раза для увеличения веса
3. **Токенизация**: приведение к нижнему регистру, удаление спецсимволов
4. **Стемминг**: русский SnowballStemmer для BM25
5. **Стоп-слова**: удалены для одного из вариантов BM25

### Источники сигналов (ранкеры)

1. **BM25Okapi** — базовый keyword search
2. **BM25Okapi (stemmed)** — keyword search с русским стеммингом
3. **BM25Okapi (stopwords)** — keyword search без стоп-слов
4. **BM25Okapi (title)** — keyword search по заголовку
5. **TF-IDF** — с биграммами по полному тексту
6. **TF-IDF (title)** — с биграммами по заголовку
7. **Sentence embeddings (MiniLM, full)** — семантический поиск
8. **Sentence embeddings (MiniLM, title)** — семантический поиск по заголовку
9. **Sentence embeddings (rubert-tiny2, full)** — русский семантический поиск
10. **Sentence embeddings (rubert-tiny2, title)** — русский семантический поиск по заголовку

### Формирование кандидатов

Для каждого запроса берётся топ-100 кандидатов через RRF из всех ранкеров. Positive — ground truth из calibration, negatives — остальные кандидаты (hard negatives).

### Фичи для LightGBM

Для каждой пары (запрос, кандидат):
- Сырые score от каждого ранкера
- Нормализованные score в [0, 1]
- Ранг кандидата в каждом ранкинге
- Длина запроса
- Длина полного текста статьи
- Длина заголовка статьи
- Interaction features между ключевыми ранкерами

Итого: 40 фичей.

### Модель ранжирования

LightGBM LambdaRank:
- `objective`: lambdarank
- `num_leaves`: 127
- `learning_rate`: 0.03
- `num_boost_round`: 300
- `feature_fraction`: 0.8
- `bagging_fraction`: 0.8
- `bagging_freq`: 5

### Embedding модели

- `paraphrase-multilingual-MiniLM-L12-v2` (~22M параметров)
- `cointegrated/rubert-tiny2` (~29M параметров)

Обе модели локальные, open-source, значительно меньше 1B параметров.

## Используемые библиотеки

- `pandas` — работа с Feather/CSV
- `beautifulsoup4` — очистка HTML
- `rank_bm25` — BM25
- `scikit-learn` — TF-IDF, KFold
- `sentence-transformers` — эмбеддинги
- `nltk` — стемминг и стоп-слова
- `lightgbm` — модель ранжирования

## Воспроизведение

```bash
python solution_v16_final.py
```

Скрипт:
1. Загружает данные
2. Строит индексы BM25/TF-IDF
3. Кодирует статьи эмбеддингами
4. Собирает обучающую выборку из calibration
5. Обучает LightGBM LambdaRank
6. Генерирует `answer.csv`

Для оценки через кросс-валидацию:

```bash
python solution_v16.py
```

## Что пробовалось ранее

- Простой RRF: MAP@10 = 0.3861
- Бинарный LightGBM: MAP@10 = 0.5787
- LambdaRank: MAP@10 = 0.5999
- Grid search + LambdaRank: MAP@10 = 0.6117
- Английский cross-encoder: ухудшил результат
- `rubert-tiny2` в одиночку: хуже MiniLM
- Обрезка body: ухудшает результат

## Ограничения

- Все модели запускаются локально
- Без внешних API и LLM >1B параметров
- Без ручной разметки тестовых данных
- Calibration используется только для обучения/валидации, не test
