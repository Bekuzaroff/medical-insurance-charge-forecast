import pandas as pd
import numpy as np


class FeatureMaker():
    """Создает новые признаки из данных страхования"""
    
    def __init__(self):
        self.fitted = False
        
    def fit(self, X, y=None):
        """Обучает FeatureMaker (ничего не делает, но нужен для совместимости)"""
        self.fitted = True
        return self
    
    def transform(self, X):
        """Применяет создание признаков к данным"""
        df = X.copy()
        
        # Базовые булевы признаки
        df["is_smoker"] = (df["smoker"] == "yes").astype(int)
        df["is_female"] = (df["sex"] == "female").astype(int)
        
        # Взаимодействия признаков
        df["age_smoker"] = df["age"] * (1 + df["is_smoker"] * 2)
        df["bmi_smoker"] = df["bmi"] * (1 + df["is_smoker"] * 1.5)
        df["age_bmi"] = df["bmi"] * df["age"] / 100  # Нормализация
        
        # BMI категории
        df["bmi_category"] = 0
        df.loc[df["bmi"] < 18.5, "bmi_category"] = 1  # Недостаточный вес
        df.loc[(df["bmi"] >= 18.5) & (df["bmi"] < 25), "bmi_category"] = 2  # Норма
        df.loc[(df["bmi"] >= 25) & (df["bmi"] < 30), "bmi_category"] = 3  # Избыточный вес
        df.loc[(df["bmi"] >= 30) & (df["bmi"] < 35), "bmi_category"] = 4  # Ожирение 1 степени
        df.loc[df["bmi"] >= 35, "bmi_category"] = 5  # Ожирение 2+ степени
       
        # Факторы риска
        df["risk_factor"] = (df["is_smoker"] * 3) + \
                           ((df["bmi"] > 30).astype(int) * 2) + \
                           ((df["age"] > 50).astype(int) * 1)
        
        # Количество детей с учетом риска
        df["many_children"] = df["children"] * (1 + df["risk_factor"] / 10)
        
        # Группы риска (сумма факторов)
        risk_features = ["age_smoker", "bmi_smoker", "age_bmi"]
        df["in_risk_group"] = df[risk_features].sum(axis=1)
        
        # Логарифмические преобразования для асимметричных признаков
        df["log_bmi"] = np.log1p(df["bmi"] - df["bmi"].min() + 1)
        df["log_age"] = np.log1p(df["age"])
        
        # Интеракции с регионами
        region_dummies = pd.get_dummies(df["region"], prefix="region")
        for region in region_dummies.columns:
            df[f"smoker_{region}"] = df["is_smoker"] * region_dummies[region]
            df[f"bmi_{region}"] = df["bmi"] * region_dummies[region]
        
        return df
    
    def fit_transform(self, X, y=None):
        """Обучает и применяет создание признаков"""
        return self.fit(X, y).transform(X)