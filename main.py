from models.model_catboost import get_model_path
from src.parser_1 import parser_pipeline
from src.load_predict import load_predict, load_cbmodel
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

class PressureModel():
    def __init__(self) -> None:
        pass

    def pipeline(self, path, is_train=False, with_normals=True) -> pd.DataFrame:
        '''
        Препроцессинг данных в структуру pd.DataFrame
        '''
        df = parser_pipeline(path, is_train=is_train, with_normals=with_normals)
        return df

    def load_predict(self, df :pd.DataFrame, geometry_type:str) -> pd.DataFrame:
        '''
        Загруза модели и предсказание для определённого числа маха и определённой геометрии.
        Модель для каждой гометрии на данный момент своя и пока доступна только сфера и crm.
        Другие модели в уже обучаются, чтобы выдавать лучший скор! Скоро мы их покажем вам!
        '''
        predict = load_predict(df,geometry_type)
        return predict

    def get_model(self,geomety_type):
        '''
        Получить модель для обучения или любых других манипуляций.
        '''
        return load_cbmodel(geomety_type)

    def save_model(self,model : CatBoostRegressor,name :str):
        '''
        Сохранение новой модели, если нужно заменить старую.
        В процессе разработки...
        '''
        model.save_model(get_model_path+'/'+name)
        return None
    
    def __call__(self,path : str) -> np.array:
        '''
        Полный пайплайн работы от получения пути файла до предсказания моделью машинного обучения давлений в каждой переданной точке.
        В пути должно встречаться одно из перечисленных категорий моделей:
        1) sphere / Sphere 2) crm 3) agard 4) luna / Luna
        all - ключ для модели обученной на всех данных. Она обладает меньшей точность, но более хорошей обобщающей способностью.
        '''
        if 'sphere' in path.lower():
            geometry_type = 'sphere'
        elif 'crm' in path.lower():
            geometry_type = 'crm'
        elif 'agard' in path.lower():
            geometry_type = 'agard'
        elif 'luna' in path.lower():
            geometry_type = 'luna'
        else:
            geometry_type = 'all'
        df = self.pipeline(path)
        predict = self.load_predict(df,geometry_type)
        return predict
    
# Пример работы класса
if __name__ == '__main__':
    model = PressureModel()
    # Путь вводится вполть до папки с числом Маха
    result = model('D:/хакатон/case_2_field_prediction-main/data_folder/Sphere_stationary/0.3M/150')
    print(result)