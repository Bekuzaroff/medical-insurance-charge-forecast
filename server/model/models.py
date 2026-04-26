import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from column_transformer import Transformer




if __name__ == "__main__":
    # LOAD DATASET, SPLITING FOR TRAIN/TEST
    print("Загрузка данных...")
    data = pd.read_csv("data/insurance.csv")
    train_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
    train_y, test_y = train_set["charges"], test_set["charges"]

    print(f"Размер обучающей выборки: {train_set.shape}")
    print(f"Размер тестовой выборки: {test_set.shape}")

    # Data transform -----------------------------
    print("\nТрансформация данных...")

    transformer = Transformer()

    train_set = train_set.drop(["charges", "region"], axis=1)
    test_set = test_set.drop(["charges", "region"], axis=1)


    train_transformed = transformer.fit_transform(train_set)
    test_transformed = transformer.transform(test_set)


    train_set[transformer.num_features + transformer.cat_features] = train_transformed
    test_set[transformer.num_features + transformer.cat_features] = test_transformed


    train_set = transformer.feature_engineering(train_set)
    test_set = transformer.feature_engineering(test_set)

    joblib.dump(train_set.columns.tolist(), "server/model/model_columns.joblib") # save all columns and columns order
    print(f"Сохранено {len(train_set.columns)} колонок: {train_set.columns.tolist()}")


    print(f"Размер после трансформации (train): {train_set.shape}")
    print(f"Размер после трансформации (test): {test_set.shape}")
    # Model training AND FIRST TEST -----------------------------
    model = RandomForestRegressor()
    model.fit(train_set, train_y)
    predicts = model.predict(test_set)

    print(mean_absolute_error(test_y, predicts))

    print(test_y[:5])
    print(predicts[:5])

    # SAVE MODEL AND TRANSFORMER
    joblib.dump(model, "server/model/best_model.joblib")
    print("модель сохранена")

    transformer_filename = "server/model/transformer.joblib"
    joblib.dump(transformer, transformer_filename)
    print(f"Трансформер сохранен как '{transformer_filename}'")

    # # EVALUTATION AND EXAMPLE OF LOADED MODEL WORK
    print("\n" + "="*60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ СОХРАНЕННОЙ МОДЕЛИ")
    print("="*60)

    # LOAD MODEL AND DATA TRANSFORMER
    loaded_model = joblib.load('server/model/best_model.joblib')
    loaded_transformer = joblib.load('server/model/transformer.joblib')

    # FIRST 3 EXAMPLE FROM TEST SET
    sample_data = test_set.head(3).copy()

    sample_predictions = loaded_model.predict(sample_data)

    sample_test_df = pd.DataFrame({
        "predicts": sample_predictions,
        "actual": test_y[:3]
    })
    print(sample_test_df)

    # def make_gr_search(model_obj, model_name, params, X, y):
    #     """Выполняет grid search с кросс-валидацией"""
    #     print(f"\nЗапуск GridSearchCV для {model_name}...")
    #     gr_s = GridSearchCV(
    #         model_obj, 
    #         params, 
    #         scoring="neg_mean_absolute_error",
    #         cv=5,
    #         n_jobs=-1,
    #         verbose=1
    #     )
    #     gr_s.fit(X, y)
        
    #     best_rmse = np.sqrt(-gr_s.best_score_)
    #     print(f"Лучшие параметры для {model_name}: {gr_s.best_params_}")
    #     print(f"Лучший MAE на кросс-валидации: {best_rmse:.4f}")
        
    #     return gr_s.best_estimator_










