# Aviahack-Gym-Bros

Переданные на флешке модели сохраняют исходное местоположение в корневой папке прокета. Получается так:
models:
 - model_catboost
    - PresureRegressor_agard.cbm
    - PresureRegressor_luna.cbm
    - PresureRegressor_crm.cbm
    - PresureRegressor_sphere.cbm
    - PresureRegressor_all.cbm
 - model_torch
    - nn_model_luna.pth
    - scaler.pkl

Чтобы запустить программу необходимо положить переданные модели на свои места. Далее вы можете использовать наш класс PressureModel как сторонний модуль. Основные функции которого:
1. Пайплайн - Создать таблицу данных исходя из пути к файлу.
2. Предикт - Выдать предикт на таблицу данных.
3. __call__ вызывает пайплайн и предикт последовательно.

Пример:
# Инициализация объекта класса
model = PressureModel()
# Получение предсказания в формате np.array()
result = model('ваш_путь_до_конкретного_файла_с_моделированием_давления/data_folder/princess_luna/0.3M/150')
