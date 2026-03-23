import pandas as pd
import joblib
from src.column_transformer import Transformer

# Загружаем обучающие данные
train_df = pd.read_csv('data/insurance.csv')

# Обучаем трансформер
transformer = Transformer()
transformer.fit(train_df)

# Сохраняем обученный трансформер
joblib.dump(transformer, 'transformer.joblib')
print("✅ Трансформер обучен и сохранен")
print(f"Количество признаков: {len(transformer.feature_names)}")
print(f"Первые 5 признаков: {transformer.feature_names[:5]}")