"""
House Prices - Advanced Regression Techniques (PySpark)

Оригинал (pandas): https://github.com/SeryVilk/house-prices-kaggle
Новая версия: https://github.com/SeryVilk/house-prices-pyspark

Запуск в PyCharm:
    всё сразу - Shift+F10 (Run 'main')
"""

# 1. Импорты и настройки

import setuptools  # noqa: F401 - distutils shim для PySpark 3.5 на Python 3.12+

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
TEST_PATH = PROJECT_ROOT / 'data' / 'raw' / 'test.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SUBMISSION_PATH = OUTPUT_DIR / 'submission.csv'
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_RAW_DIR.mkdir(parents = True, exist_ok = True)
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

RANDOM_STATE = 42
CV_FOLDS = 5

spark = (
    SparkSession.builder
    .appName('house-prices-pyspark')
    .master('local[4]')
    .config('spark.sql.shuffle.partitions', '4')
    .config('spark.driver.memory', '4g')
    .getOrCreate()
)
spark.sparkContext.setLogLevel('WARN')

_log4j = spark._jvm.org.apache.logging.log4j
_log4j.LogManager.getLogger('org.apache.spark.scheduler.DAGScheduler').setLevel(_log4j.Level.ERROR)
_log4j.LogManager.getLogger('org.apache.spark.sql.catalyst.util.SparkStringUtils').setLevel(_log4j.Level.ERROR)

print('Spark:', spark.version)


# 2. Загрузка данных

# Train - 1460 объектов, Test - 1459. Признаки: 79 объясняющих переменных + 'Id' + таргет 'SalePrice'

train = spark.read.csv(str(TRAIN_PATH), header=True, inferSchema=True)
test = spark.read.csv(str(TEST_PATH),  header=True, inferSchema=True)

# 'MiscFeature' - 96% пропусков, отбрасываем как в pandas-версии
train = train.drop('MiscFeature')
test = test.drop('MiscFeature')

print('Train:', train.count(), 'rows,', len(train.columns), 'cols')
print('Test: ', test.count(), 'rows,', len(test.columns), 'cols')

train.cache()
test.cache()

# 'inferSchema' иногда ставит StringType на колонки с большим числом пропусков
numeric_cast = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                'TotalBsmtSF', 'GarageCars', 'GarageArea', 'GarageYrBlt',
                'BsmtFullBath', 'BsmtHalfBath']
for c in numeric_cast:
    train = train.withColumn(c, F.col(c).cast('double'))
    test = test.withColumn(c, F.col(c).cast('double'))


# 3. EDA (краткий)

# Таргет 'SalePrice' имеет правостороннюю асимметрию => логарифмируем
# Топ-корреляции с таргетом: 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF'

train.select('SalePrice').describe().show()

numeric_for_corr = [c for c, t in train.dtypes if t in ('int', 'double', 'bigint') and c != 'Id']
corr = train.select(numeric_for_corr).toPandas().corr()['SalePrice'].drop('SalePrice')
print('Top correlations with SalePrice:')
print(corr.sort_values(ascending=False).head(10))


# 4. Предобработка
# 4.1. Заполнение пропусков

# Числовые: "нет объекта" => 0
zero_numeric = ['MasVnrArea', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath',
                'GarageCars', 'GarageArea', 'GarageYrBlt']
train = train.fillna(0, subset = zero_numeric)
test = test.fillna(0, subset = zero_numeric)

# Числовые: медиана по train-части (без утечки из test)
median_numeric = ['LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF']
for c in median_numeric:
    median_val = train.approxQuantile(c, [0.5], 0.001)[0]
    train = train.fillna(median_val, subset = [c])
    test = test.fillna(median_val, subset = [c])

# Категориальные: "нет объекта" => 'No'
no_cat = ['Alley', 'MasVnrType', 'GarageType', 'GarageFinish', 'Fence']
train = train.fillna('No', subset = no_cat)
test = test.fillna('No', subset = no_cat)

# Категориальные: мода по train-части
def get_mode(df, col):
    return df.groupBy(col).count().orderBy(F.desc('count')).limit(1).collect()[0][0]

mode_cat = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
            'Functional', 'SaleType', 'Electrical']
mode_vals = {c: get_mode(train, c) for c in mode_cat}
train = train.fillna(mode_vals)
test = test.fillna(mode_vals)


# 4.2. Ordinal-кодирование

# Качественные признаки с естественным порядком ('Ex > Gd > TA > Fa > Po') переводим в числа

mapping_qual = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
mapping_exposure = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
mapping_fin = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}

qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
exposure_cols = ['BsmtExposure']
fin_cols = ['BsmtFinType1', 'BsmtFinType2']


def ordinal_encode(df, col, mapping, null_value):
    """Строки -> числа по порядку. NULL превращается в null_value"""
    expr = F.lit(float(null_value))
    for key, val in mapping.items():
        expr = F.when(F.col(col) == key, F.lit(float(val))).otherwise(expr)
    return df.withColumn(col, expr)


for c in qual_cols:
    train = ordinal_encode(train, c, mapping_qual, 0)
    test = ordinal_encode(test, c, mapping_qual, 0)

for c in exposure_cols:
    train = ordinal_encode(train, c, mapping_exposure, -1)
    test = ordinal_encode(test, c, mapping_exposure, -1)

for c in fin_cols:
    train = ordinal_encode(train, c, mapping_fin, -1)
    test = ordinal_encode(test, c, mapping_fin, -1)


# 4.3. ML Pipeline - препроцессинг

# Собираем все стадии препроцессинга в отдельный 'Pipeline' и фитим его ОДИН раз
# У 'StringIndexer' / 'OneHotEncoder' / 'VectorAssembler' нет гиперпараметров

train = train.withColumn('logSalePrice', F.log1p(F.col('SalePrice')))

# Оставшиеся строки - номинальные категории
categorical_cols = [c for c, t in train.dtypes if t == 'string']

# Все числовые, кроме Id и таргетов. Ordinal-признаки после кодирования тоже сюда попадают
numeric_cols = [c for c, t in train.dtypes
                if t in ('int', 'double', 'bigint')
                and c not in ('Id', 'SalePrice', 'logSalePrice')]

print(f'Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols)}')

indexers = [StringIndexer(inputCol = c, outputCol = f'{c}_idx', handleInvalid = 'keep') for c in categorical_cols]
encoders = [OneHotEncoder(inputCol = f'{c}_idx', outputCol = f'{c}_ohe', dropLast = True) for c in categorical_cols]

feature_cols = numeric_cols + [f'{c}_ohe' for c in categorical_cols]
assembler = VectorAssembler(inputCols = feature_cols, outputCol = 'features', handleInvalid = 'skip')

preprocessing = Pipeline(stages = indexers + encoders + [assembler])
preprocessing_model = preprocessing.fit(train)

train_prep = preprocessing_model.transform(train).cache()
test_prep = preprocessing_model.transform(test).cache()

# Проверка: ни одна строка не должна потеряться на 'VectorAssembler'
print('Train after prep:', train_prep.count(), 'rows (expected 1460)')
print('Test after prep: ', test_prep.count(), 'rows (expected 1459)')


# 5. Подготовка X / y

# Отдельные X_train / y_train не нужны: препроцессинг уже собран в 'features', а таргетом служит колонка 'logSalePrice'

print('Train shape:', (train_prep.count(), len(train_prep.columns)))
print('Test shape: ', (test_prep.count(), len(test_prep.columns)))


# 6. Обучение моделей

# Стратегия: 'CrossValidator' + 'ParamGridBuilder' только над GBT
# Препроцессинг уже зафичен -> каждая итерация тренирует только модель

evaluator = RegressionEvaluator(
    labelCol = 'logSalePrice',
    predictionCol = 'prediction',
    metricName = 'rmse',
)

gbt = GBTRegressor(
    featuresCol = 'features',
    labelCol = 'logSalePrice',
    predictionCol = 'prediction',
    seed = RANDOM_STATE,
    maxIter = 200,
    maxDepth = 5,
    stepSize = 0.1,
    subsamplingRate = 0.8,
)

param_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [5, 7])
    .addGrid(gbt.stepSize, [0.05, 0.1])
    .build()
)
print(f'CV grid: {len(param_grid)} combos x {CV_FOLDS} folds = {len(param_grid) * CV_FOLDS} fits')

cv = CrossValidator(
    estimator = gbt,
    estimatorParamMaps = param_grid,
    evaluator = evaluator,
    numFolds = CV_FOLDS,
    seed = RANDOM_STATE,
    parallelism = 1,
)

cv_model = cv.fit(train_prep)

print(f'\nBest CV RMSE (log): {min(cv_model.avgMetrics):.5f}')
best_gbt = cv_model.bestModel
print(f'Best params: maxDepth = {best_gbt.getMaxDepth()}, '
      f'stepSize = {best_gbt.getStepSize()}')


# 7. Формирование submission

# Таргет обучался на log1p - для итоговых цен делаем expm1
# 'test_prep' уже содержит 'features'

preds = (
    cv_model.transform(test_prep)
    .select(
        F.col('Id').cast('int').alias('Id'),
        F.expm1(F.col('prediction')).alias('SalePrice'),
    )
    .orderBy('Id')
)

OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
preds.toPandas().to_csv(str(SUBMISSION_PATH), index = False)
print('Saved ->', SUBMISSION_PATH)


# 8. Выводы

print('''
8. Выводы

- PySpark DataFrame API вместо pandas на всех этапах
- Препроцессинг (StringIndexer + OneHotEncoder + VectorAssembler) вынесен в отдельный Pipeline
  и фитится один раз - 'CrossValidator' тренирует только GBT
- CrossValidator + ParamGridBuilder подбирают гиперпараметры с учётом фолдов
- Таргет логарифмирован (log1p), обратное преобразование - expm1
- Пропуски заполнены с учётом семантики: медиана/мода по train-части, 'No' = отсутствие объекта.

Возможные улучшения:
- Feature engineering
- Ансамбль: GBT + RandomForestRegressor + LinearRegression
''')

train.unpersist()
train_prep.unpersist()
test_prep.unpersist()
spark.stop()