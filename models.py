
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.column_transformer import Transformer

train_set, test_set = train_test_split(pd.read_csv("data\insurance.csv"), test_size=0.1)
train_y, test_y = train_set["charges"], test_set["charges"]

train_set = train_set.drop("charges", axis=1)
test_set = test_set.drop("charges", axis=1)

transformer = Transformer()

train_prepared = transformer.fit_transform(train_set)
test_prepared = transformer.fit_transform(test_set)

def compare_models(models: dict, metric: str = "test_rmse"):
    results = []
    for name in models.keys():
        model = models[name]
        model.fit(train_prepared, train_y)
        
        train_predicts = model.predict(train_prepared)
        test_predicts = model.predict(test_prepared)

        train_table = pd.DataFrame({"charges": train_y.iloc[:5], 
                                    "predicts": train_predicts[:5]})
        
        test_table = pd.DataFrame({"charges": test_y.iloc[:5], 
                                    "predicts": test_predicts[:5]})
        
        print(train_table)
        print(test_table)

        model = {"model": name, 
                  "train_rmse": np.sqrt(mean_squared_error(train_y, train_predicts)), 
                  "test_rmse": np.sqrt(mean_squared_error(test_y, test_predicts)),
                  "train_mse": mean_squared_error(train_y, train_predicts), 
                  "test_mse": mean_squared_error(test_y, test_predicts),
                  "train_mae": mean_absolute_error(train_y, train_predicts), 
                  "test_mae": mean_absolute_error(test_y, test_predicts),
                  "model_object": models[name]}
        results.append(model)
    
    results_df = pd.DataFrame(results)

    if metric == "test_rmse":
        results_df = results_df.sort_values(metric)
    return results_df


models = {"Linear Regression": LinearRegression(), 
          "SVR": SVR(),
          "Desicion tree": DecisionTreeRegressor(),
          "Random forest": RandomForestRegressor(max_depth=50, max_features="sqrt")}
model_df = compare_models(models)
print(model_df)
best_model = model_df.iloc[0]
print("best model: ", best_model["model"])

joblib.dump(best_model["model_object"], "best_model.joblib")
print("model saved")



