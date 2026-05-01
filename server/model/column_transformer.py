import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class Transformer():
    def __init__(self):
        self.num_features = ["age", "bmi", "children"]
        self.cat_features = ["sex", "smoker"]
        
    def fit(self, X):
        df = X.copy()
        # monitoring
        print("------ monitoring -------------------")
        print("HEAD")
        print(df.head())
        print("DESCRIBE")
        print(df.describe())
        print("INFO")
        print(df.info())
        ("------------------------------------")

        possible_categories = {
            'sex': ['female', 'male'],
            'smoker': ['no', 'yes']
        }


        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        cat_pipeline = Pipeline([
             ("cat_imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("enc", OrdinalEncoder(categories=[possible_categories['sex'], possible_categories['smoker']],
                handle_unknown='use_encoded_value',
                unknown_value=-1))
        ])
        self.col_trans = ColumnTransformer([
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        self.col_trans.fit(df)
        



    
    def transform(self, X: pd.DataFrame):
        return self.col_trans.transform(X)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def feature_engineering(self, X: pd.DataFrame):
        df = X.copy()

        bmi_conditions = [
            df['bmi'] < 18.5,
            (df['bmi'] >= 18.5) & (df['bmi'] < 25),
            (df['bmi'] >= 25) & (df['bmi'] < 30),
            df['bmi'] >= 30
        ]
        
        bmi_choices = [3, 1, 7, 10]
        
        df['bmi_category'] = np.select(bmi_conditions, bmi_choices, default=0)
        

        age_conditions = [
            df['age'] < 22,
            (df['age'] >= 22) & (df['age'] < 35),
            (df['age'] >= 35) & (df['age'] < 60),
            df['age'] >= 60
        ]
        age_choices = [1, 5, 10, 15]
        df['age_category'] = np.select(age_conditions, age_choices, default=0)

        children_conditions = [
            df['children'] == 0,
            (df['children'] >= 1) & (df['children'] < 3),
            (df['children'] >= 3) & (df['children'] < 6),
            df['children'] >= 6
        ]
        children_choices = [1, 5, 10, 15]
        df['children_category'] = np.select(children_conditions, children_choices, default=0)
        
        df['BMI_x_Smoker'] = df['bmi'] * df['smoker']  # убрал сравнение
        df['Smoker_x_Age'] = df['smoker'] * df['age']  # убрал сравнение
        df['Smoker_x_BMI_x_Age'] = df['smoker'] * df['bmi'] * df['age']  # убрал сравнение
        df['BMI_x_Children'] = df['bmi'] * df['children']  # ожирение + дети
        df['Smoker_x_Age'] = (df['smoker'] == 1) * df['age']  # курящие пожилые
        df['Smoker_x_BMI_x_Age'] = (df['smoker'] == 1) * df['bmi'] * df['age']  # тройное

        # Квадраты и степени
        df['age_squared'] = df['age'] ** 2  # риск растет экспоненциально с возрастом
        df['bmi_squared'] = df['bmi'] ** 2  # ожирение имеет пороговый эффект
        df['age_cubic'] = df['age'] ** 3  # для捕捉сильных нелинейностей

        # Логарифмы (для признаков с длинным хвостом)
        df['log_bmi'] = np.log1p(df['bmi'] - 15)  # нормализация ИМТ
        df['log_age'] = np.log(df['age'] + 1)

        # Extreme BMI (очень низкий/высокий)
        df['obese'] = (df['bmi'] >= 30).astype(int)
        df['underweight'] = (df['bmi'] <= 18.5).astype(int)
        df['extreme_bmi'] = ((df['bmi'] < 18.5) | (df['bmi'] > 35)).astype(int)

        # Пожилые с детьми
        df['elderly_with_children'] = ((df['age'] > 50) & (df['children'] > 0)).astype(int)

        # Молодые курильщики
        df['young_smoker'] = ((df['age'] < 30) & (df['smoker'] == 1)).astype(int)

        
        return df
        
        

        


        