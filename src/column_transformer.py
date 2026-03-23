import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from .feature_maker import FeatureMaker


class Transformer():
    def __init__(self):
        self.pipeline = None
        self.num_attrs = ["age", "bmi", "age_smoker", "bmi_smoker", "age_bmi", "risk_factor", 
                          "in_risk_group", "many_children", "bmi_category", "log_bmi", "log_age"]
        self.cat_attrs = ["sex", "smoker"]  # region убран
        self.feature_maker = FeatureMaker()
        self.feature_names = None
        
    def fit(self, X, y=None):
        df = X.copy()
        df = self.feature_maker.fit_transform(df)
        
        available_num = [col for col in self.num_attrs if col in df.columns]
        available_cat = [col for col in self.cat_attrs if col in df.columns]
        
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler())
        ])
        
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("o_encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        self.pipeline = ColumnTransformer([
            ("num", num_transformer, available_num),
            ("cat", cat_transformer, available_cat)
        ], remainder='drop')
        
        self.pipeline.fit(df[available_num + available_cat])
        
        self.used_num_attrs = available_num
        self.used_cat_attrs = available_cat
        
        transformed = self.pipeline.transform(df)
        base_features = self.used_num_attrs + self.used_cat_attrs
        base_df = pd.DataFrame(transformed, columns=base_features)
        
        result_df = base_df.copy()
        
        for attr in self.used_num_attrs:
            result_df[f"{attr}**2"] = base_df[attr] ** 2
        
        for i in range(len(base_features)):
            for j in range(i + 1, len(base_features)):
                result_df[f"{base_features[i]}X{base_features[j]}"] = \
                    base_df[base_features[i]] * base_df[base_features[j]]
        
        self.feature_names = list(result_df.columns)
        return self
    
    def transform(self, X: pd.DataFrame):
        df = X.copy()
        df = self.feature_maker.transform(df)
        
        transformed = self.pipeline.transform(df)
        base_features = self.used_num_attrs + self.used_cat_attrs
        base_df = pd.DataFrame(transformed, columns=base_features, index=df.index)
        
        result_df = base_df.copy()
        
        for attr in self.used_num_attrs:
            result_df[f"{attr}**2"] = base_df[attr] ** 2
        
        for i in range(len(base_features)):
            for j in range(i + 1, len(base_features)):
                result_df[f"{base_features[i]}X{base_features[j]}"] = \
                    base_df[base_features[i]] * base_df[base_features[j]]
        
        if self.feature_names is not None:
            result_df = result_df[self.feature_names]
        
        return result_df
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)