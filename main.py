import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.analys.analys import Analys
from src.column_transformer import Transformer
from src.feature_maker import FeatureMaker






def main():
    # Загружаем данные
    data = pd.read_csv("data/insurance.csv")
    
    print("Исходные данные:")
    print(data.head())
    print(f"\nРазмер данных: {data.shape}")
    print(f"Колонки: {data.columns.tolist()}")
    
    # Разделяем на train/test
    train_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
    
    print(f"\nTrain размер: {train_set.shape}")
    print(f"Test размер: {test_set.shape}")
    
    
    # Тестируем Transformer
    print("\n" + "="*50)
    print("Тестируем Transformer:")
    print("="*50)
    
    transformer = Transformer()
    train_transformed = transformer.fit_transform(train_set)
    
    print("\nПреобразованные данные:")
    print(train_transformed.head())
    print(f"\nРазмер после преобразования: {train_transformed.shape}")
    print(f"Имена признаков: {train_transformed.columns.tolist()}")
    
    # Анализ корреляций
    print("\n" + "="*50)
    print("Анализ корреляций:")
    print("="*50)
    
    if 'charges' in train_transformed.columns:
        corr_m = train_transformed.corr()
        print("\nКорреляция с charges (топ-10):")
        charges_corr = corr_m['charges'].sort_values(ascending=False)
        print(charges_corr.head(10))
        
        # Визуализация
        print("\nСоздаем графики...")
        fig, axes = Analys.scatter_target_corr(train_transformed, corr_m)
        
        # Дополнительная визуализация
        fig2, ax2 = Analys.plot_correlation_matrix(train_transformed)
        
        plt.show()
    else:
        print("Колонка 'charges' не найдена в преобразованных данных!")
    
   


if __name__ == '__main__':
    main()
    