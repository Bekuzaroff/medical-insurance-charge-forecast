import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


from .feature_maker import FeatureMaker


class Transformer():
    """Трансформер для предобработки данных"""
    
    def __init__(self):
        pass
    
    def fit(self):
        return self
    

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        
        num_attrs = ["age", "bmi",   
                     "age_smoker", "bmi_smoker", "age_bmi",  "risk_factor", 
                     "in_risk_group", "many_children"]
        cat_attrs = ["sex", "smoker", "region"]

        num_trans = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler())
        ])
        cat_trans = Pipeline([
            ("o_encoder", OrdinalEncoder())
        ])

        transformer = ColumnTransformer([
            ("num", num_trans, num_attrs),
            ("cat", cat_trans, cat_attrs)
        ])

        main_pipeline = Pipeline([
            ("feature", FeatureMaker()),
            ("preproccessor", transformer)
        ])

        all_attrs = []

        all_attrs.extend(num_attrs)
        all_attrs.extend(cat_attrs)

        df[all_attrs] = main_pipeline.fit_transform(df)

        df = df.drop("sex", axis=1)
        df = df.drop("region", axis=1)
        df = df.drop("bmi", axis=1)
        df = df.drop("bmi_smoker", axis=1)
        df = df.drop("children", axis=1)
        
        return df
    
    def fit_transform(self, X):
        return self.fit().transform(X)