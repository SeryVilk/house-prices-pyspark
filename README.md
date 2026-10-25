# house-prices-pyspark

Решение соревнования [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) на **Apache Spark**.

Задача: по 79 признакам предсказать стоимость жилого дома. Метрика - **RMSE**.

Переписал свой [pandas-ноутбук](https://github.com/SeryVilk/house-prices-kaggle) на PySpark **DataFrame API + ML Pipelines** - с акцентом на идиоматичный Spark, а не на `toPandas()` в каждой строке.

## Использованные приёмы PySpark

- `SparkSession` с настройкой `shuffle.partitions`, `driver.memory` и кастомными log4j-фильтрами
- `DataFrame` API: `withColumn`, `fillna`, `drop`, `when/otherwise`, `approxQuantile`
- Явный `cast` колонок, на которых `inferSchema` ошибся (например `LotFrontage`)
- Ordinal-кодирование через цепочку `F.when` вместо `map` из pandas
- ML `Pipeline`: `StringIndexer` -> `OneHotEncoder` -> `VectorAssembler`
- Препроцессинг вынесен в отдельный `Pipeline`, фитится **один раз** - `CrossValidator` тренирует только модель (ускорение CV в ~5 раз)
- `GBTRegressor` из `pyspark.ml.regression`
- `CrossValidator` + `ParamGridBuilder` для подбора гиперпараметров
- `RegressionEvaluator` (RMSE)
- `log1p` / `expm1` для таргета
- `cache()` / `unpersist()` для контроля памяти
- `toPandas()` — только один раз, для сохранения финального `submission.csv`

Весь код — в одном файле `main.py`, разбитом на секции `# 1`, `# 2`, ... `# 8`. Можно запускать в PyCharm как единый скрипт (**Shift+F10**) или как ячейки

Данные необходимо скачать с Kaggle самостоятельно