

class FeatureMaker():
    """Создает новые признаки из данных страхования"""
    
    def __init__(self, poly_features=False):
        self.poly_features = poly_features
    
    def fit(self):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Базовые булевы признаки
        df["is_smoker"] = (df["smoker"] == "yes").astype(int)
        df["is_female"] = (df["sex"] == "female").astype(int)
        
        # Взаимодействия признаков
        df["age_smoker"] = 4 * df["age"] + df["is_smoker"] * 5
        df["bmi_smoker"] = 5 * df["bmi"] + df["is_smoker"] * 7
        df["age_bmi"] = df["bmi"] * df["age"]
        
        # BMI категории
        df["normal_bmi"] = ((df["bmi"] > 25) & (df["bmi"] < 35)).astype(int)
       
        # Факторы риска
        df["risk_factor"] = (df["is_smoker"] * 7) + \
                           ((df["bmi"] > 30).astype(int) * 1) + \
                           ((df["age"] > 55).astype(int) * 2)
        
        df["many_children"] = (df["risk_factor"] / (df["children"]  +  1) * 6)
        
        
        # Группы риска
        df["in_risk_group"] = df[["age_smoker", "age_bmi", "bmi_smoker"]].sum(axis=1)
                             
        
        # Полиномиальные признаки
        if self.poly_features:
            df["age_squared"] = df["age"] ** 2
            df["bmi_squared"] = df["bmi"] ** 2
        
        return df
    
    def fit_transform(self, X, y=None):
        return self.fit().transform(X)