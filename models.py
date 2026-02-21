import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.column_transformer import Transformer

# Загрузка и разделение данных
print("Загрузка данных...")
data = pd.read_csv("data/insurance.csv")
train_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
train_y, test_y = train_set["charges"], test_set["charges"]

print(f"Размер обучающей выборки: {train_set.shape}")
print(f"Размер тестовой выборки: {test_set.shape}")

# Трансформация данных
print("\nТрансформация данных...")
transformer = Transformer()


train_prepared = transformer.fit_transform(train_set)
test_prepared = transformer.transform(test_set)

print(f"Размер после трансформации (train): {train_prepared.shape}")
print(f"Размер после трансформации (test): {test_prepared.shape}")

# Отбор признаков по корреляции с целевой переменной
print("\nОтбор признаков по корреляции...")

# Добавляем целевую переменную для расчета корреляции
train_full = train_prepared.copy()
train_full["charges"] = train_y

# Сортируем признаки по корреляции с charges
corr1 = train_full.corr()["charges"].sort_values(ascending=False)
print(f"Топ-5 признаков по корреляции:\n{corr1.head(6)}")  # 6 включает charges

# Выбираем топ-70 признаков (исключая саму charges)
top_features = corr1.index[1:71]  # пропускаем первый элемент (charges)
train_prepared = train_prepared[top_features]

# Для тестовых данных используем те же признаки
test_prepared = test_prepared[top_features]

print(f"Размер после отбора признаков (train): {train_prepared.shape}")
print(f"Размер после отбора признаков (test): {test_prepared.shape}")

def make_gr_search(model_obj, model_name, params, X, y):
    """Выполняет grid search с кросс-валидацией"""
    print(f"\nЗапуск GridSearchCV для {model_name}...")
    gr_s = GridSearchCV(
        model_obj, 
        params, 
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    gr_s.fit(X, y)
    
    best_rmse = np.sqrt(-gr_s.best_score_)
    print(f"Лучшие параметры для {model_name}: {gr_s.best_params_}")
    print(f"Лучший RMSE на кросс-валидации: {best_rmse:.4f}")
    
    return gr_s.best_estimator_

def compare_models(models: dict, X_train, y_train, X_test, y_test, metric: str = "test_rmse"):
    """Сравнивает различные модели регрессии"""
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Обучение модели: {name}")
        print(f"{'='*50}")
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказания
        train_predicts = model.predict(X_train)
        test_predicts = model.predict(X_test)
        
        # Метрики
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predicts))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predicts))
        train_r2 = r2_score(y_train, train_predicts)
        test_r2 = r2_score(y_test, test_predicts)
        
        # Пример предсказаний
        train_sample = pd.DataFrame({
            "Фактические": y_train.iloc[:5].values,
            "Предсказанные": train_predicts[:5].round(2)
        })
        test_sample = pd.DataFrame({
            "Фактические": y_test.iloc[:5].values,
            "Предсказанные": test_predicts[:5].round(2)
        })
        
        print(f"\nПример предсказаний на обучающей выборке:\n{train_sample}")
        print(f"\nПример предсказаний на тестовой выборке:\n{test_sample}")
        print(f"\nRMSE на обучении: {train_rmse:.4f}")
        print(f"RMSE на тесте: {test_rmse:.4f}")
        print(f"R² на обучении: {train_r2:.4f}")
        print(f"R² на тесте: {test_r2:.4f}")
        
        # Кросс-валидация для оценки стабильности
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            scoring="neg_mean_squared_error", 
            cv=5, n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(-cv_scores).std()
        print(f"CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")

        # Сохраняем результаты
        results.append({
            "model": name,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "cv_rmse": cv_rmse,
            "cv_std": cv_std,
            "train_mse": mean_squared_error(y_train, train_predicts),
            "test_mse": mean_squared_error(y_test, test_predicts),
            "train_mae": mean_absolute_error(y_train, train_predicts),
            "test_mae": mean_absolute_error(y_test, test_predicts),
            "model_object": model
        })
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Сортируем по выбранной метрике
    if metric in results_df.columns:
        results_df = results_df.sort_values(metric)
    
    return results_df

# Определяем модели для сравнения
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
    "Random Forest": RandomForestRegressor(
        random_state=42, 
        n_estimators=100,
        max_depth=15,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        verbose=-1
    ),
    "Stacking": StackingRegressor(
        estimators=[
            ("xgb", XGBRegressor(random_state=42, n_estimators=50)),
            ("lgbm", LGBMRegressor(random_state=42, n_estimators=50, verbose=-1)),
            ("rf", RandomForestRegressor(random_state=42, n_estimators=50))
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    )
}

# Сравниваем модели
print("\n" + "="*60)
print("НАЧАЛО СРАВНЕНИЯ МОДЕЛЕЙ")
print("="*60)

model_df = compare_models(
    models, 
    train_prepared, 
    train_y, 
    test_prepared, 
    test_y
)

# Выводим результаты
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕЛЕЙ")
print("="*60)
print("\nСортировка по тестовому RMSE:")
print(model_df[["model", "train_rmse", "test_rmse", "train_r2", "test_r2", "cv_rmse"]].round(4))

# Выбираем лучшую модель
best_model_info = model_df.iloc[0]
print(f"\n{'!'*50}")
print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_info['model']}")
print(f"Тестовый RMSE: {best_model_info['test_rmse']:.4f}")
print(f"Тестовый R²: {best_model_info['test_r2']:.4f}")
print(f"CV RMSE: {best_model_info['cv_rmse']:.4f} (+/- {best_model_info['cv_std']:.4f})")
print(f"{'!'*50}")


# if best_model_info['model'] == "Random Forest":
#     print("\n" + "="*60)
#     print("ЗАПУСК GRID SEARCH ДЛЯ RANDOM FOREST")
#     print("="*60)
    
#     params = {
#         "n_estimators": [100, 200, 300],
#         "max_depth": [10, 15, 20, None],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
#         "max_features": ["sqrt", "log2"]
#     }
    
#     best_rf_model = make_gr_search(
#         RandomForestRegressor(random_state=42, n_jobs=-1),
#         "Random Forest",
#         params,
#         train_prepared,
#         train_y
#     )
    
#     # Проверяем улучшенную модель
#     test_predicts = best_rf_model.predict(test_prepared)
#     improved_rmse = np.sqrt(mean_squared_error(test_y, test_predicts))
#     print(f"\nУлучшенный RMSE на тесте: {improved_rmse:.4f}")
    
#     if improved_rmse < best_model_info['test_rmse']:
#         print("Grid Search улучшил модель!")
#         best_model = best_rf_model
#     else:
#         print("Grid Search не улучшил модель, сохраняем оригинал")
#         best_model = best_model_info["model_object"]
# else:
#     best_model = best_model_info["model_object"]

# Сохраняем лучшую модель
best_model = best_model_info["model_object"]
model_filename = "best_model.joblib"
joblib.dump(best_model, model_filename)
print(f"\n✓ Модель сохранена как '{model_filename}'")

# Сохраняем также трансформер для использования в продакшене
transformer_filename = "transformer.joblib"
joblib.dump(transformer, transformer_filename)
print(f"✓ Трансформер сохранен как '{transformer_filename}'")

# Пример использования сохраненной модели
print("\n" + "="*60)
print("ПРИМЕР ИСПОЛЬЗОВАНИЯ СОХРАНЕННОЙ МОДЕЛИ")
print("="*60)

# Загружаем модель и трансформер
loaded_model = joblib.load(model_filename)
loaded_transformer = joblib.load(transformer_filename)

# Берем первые 3 примера из тестовой выборки
sample_data = test_set.head(3).copy()
sample_actual = sample_data["charges"].values

# Трансформируем и предсказываем
sample_prepared = loaded_transformer.transform(sample_data)
sample_prepared = sample_prepared[top_features]  # Используем те же признаки
sample_predictions = loaded_model.predict(sample_prepared)

# Выводим результаты
example_df = pd.DataFrame({
    "Фактические": sample_actual,
    "Предсказанные": sample_predictions.round(2),
    "Ошибка": (sample_actual - sample_predictions).round(2),
    "Ошибка %": ((np.abs(sample_actual - sample_predictions) / sample_actual * 100).round(2))
})
print("\nПримеры предсказаний на новых данных:")
print(example_df.to_string(index=False))

print(f"\nСредняя абсолютная ошибка на примерах: {example_df['Ошибка'].abs().mean():.2f}")
print(f"Средняя относительная ошибка: {example_df['Ошибка %'].mean():.2f}%")