from models.model_catboost import get_model_path
from functools import cache

#import torch
#from torch import nn
#import pickle as pkl

import pandas as pd
from catboost import CatBoostRegressor, Pool

@cache
def load_cbmodel(geomety_type):
    model = CatBoostRegressor()
    if not get_model_path(geomety_type) is None:
        model.load_model(get_model_path(geomety_type))
        return model
    else:
        return None
    

# @cache
# def load_torch_model(geomety_type):
#     model = None
#     if not get_model_path(geomety_type) is None:
#         model = torch.load(get_model_path(geomety_type))
#         return model
#     else:
#         return None
#     pass


def load_predict(df : pd.DataFrame, geometry_type :str) -> pd.Series:
    # if not load_torch_model(geometry_type) is None:        
    #     model = nn.Sequential(
    #         nn.Linear(7, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 1),
    #     )
    #   model = load_torch_model(geometry_type)
    #with open('/model/scaler_luna.pkl','rb') as file:
    #    scaler = pkl.load(file)

    #else
    model = load_cbmodel(geometry_type)
    

    #df[['Ux', 'P']] = scaler.transform(df[['Ux','P']])
    input_pool = Pool(df)
    predict = model.predict(input_pool)
    return predict