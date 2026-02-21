import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from .feature_maker import FeatureMaker


class Transformer():
    """Трансформер для предобработки данных"""
    
    def __init__(self):
        self.pipeline = None
        self.num_attrs = ["age", "bmi", "age_smoker", "bmi_smoker", "age_bmi", "risk_factor", 
                          "in_risk_group", "many_children"]
        self.cat_attrs = ["sex", "smoker", "region"]
        self.all_attrs = self.num_attrs + self.cat_attrs
        self.feature_maker = FeatureMaker()
        
    def fit(self, X, y=None):
        """Обучает трансформер на обучающих данных"""
        df = X.copy()
        
        # Создаем новые признаки
        df = self.feature_maker.fit_transform(df)
        
        # Проверяем, что все необходимые колонки существуют
        missing_num = [col for col in self.num_attrs if col not in df.columns]
        if missing_num:
            print(f"Предупреждение: отсутствуют числовые колонки {missing_num}")
        
        missing_cat = [col for col in self.cat_attrs if col not in df.columns]
        if missing_cat:
            print(f"Предупреждение: отсутствуют категориальные колонки {missing_cat}")
        
        # Создаем пайплайн для предобработки
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler())
        ])
        
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("o_encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Создаем ColumnTransformer с доступными колонками
        available_num = [col for col in self.num_attrs if col in df.columns]
        available_cat = [col for col in self.cat_attrs if col in df.columns]
        
        self.pipeline = ColumnTransformer([
            ("num", num_transformer, available_num),
            ("cat", cat_transformer, available_cat)
        ], remainder='drop')
        
        # Обучаем пайплайн
        self.pipeline.fit(df[available_num + available_cat])
        
        # Сохраняем использованные колонки
        self.used_num_attrs = available_num
        self.used_cat_attrs = available_cat
        self.used_all_attrs = available_num + available_cat
        
        return self
    
    def transform(self, X: pd.DataFrame):
        """Применяет трансформер к данным"""
        df = X.copy()
        
        # Создаем новые признаки
        df = self.feature_maker.transform(df)
        
        # Проверяем, что пайплайн обучен
        if self.pipeline is None:
            raise ValueError("Трансформер не обучен. Сначала вызовите fit() или fit_transform()")
        
        # Применяем пайплайн
        transformed = self.pipeline.transform(df)
        
        # Создаем DataFrame с правильными именами колонок
        feature_names = self.used_num_attrs + self.used_cat_attrs
        result_df = pd.DataFrame(
            transformed, 
            columns=feature_names,
            index=df.index
        )
        
        # Добавляем полиномиальные признаки (квадраты)
        for attr in self.used_num_attrs:
            result_df[f"{attr}**2"] = result_df[attr] ** 2
        
        # Добавляем взаимодействия признаков
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):  # Избегаем дублирования
                result_df[f"{feature_names[i]}X{feature_names[j]}"] = \
                    result_df[feature_names[i]] * result_df[feature_names[j]]
        
        return result_df
    
    def fit_transform(self, X, y=None):
        """Обучает трансформер и применяет к данным"""
        return self.fit(X, y).transform(X)