from models.model_catboost import get_model_path
from functools import cache
import pandas as pd
from catboost import CatBoostRegressor, Pool

@cache
def load_model(geomety_type):
    model = CatBoostRegressor()
    print(f'Тип полученной геометрии: {geomety_type}')
    model.load_model(get_model_path(geomety_type))
    return model

def load_predict(df : pd.DataFrame, geometry_type :str) -> pd.Series:
    model = load_model(geometry_type)
    input_pool = Pool(df)
    predict = model.predict(input_pool)
    return predict