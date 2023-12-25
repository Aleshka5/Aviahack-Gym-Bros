# Aviahack-Gym-Bros 
![image](https://github.com/Aleshka5/Aviahack-Gym-Bros/assets/78702396/63205cb3-4ed3-4e7b-a6d9-78f366cdbe1e)
# Введение

# Проблематика
Одним из кейсов этого хакатона стало создание модели для предсказания силы давления ветра при продувке объекта в аэродинамической трубе. Датасет, который нам предоставили имел следующие парамметры для каждой 3Д модели:

- Для каждого полигона модели
  - Координаты середины полигона    
- Глобальные параметры
  - Скорость ветра
  - Направление ветра

Всего у нас было 4 модели:
<p><img src="https://github.com/Aleshka5/Aviahack-Gym-Bros/blob/normal-gpu-calculating/gifs/sphere_animation.gif" width="255" height="255" alt="lorem">
<img src="https://github.com/Aleshka5/Aviahack-Gym-Bros/blob/normal-gpu-calculating/gifs/crm_animation.gif" width="255" height="255" alt="lorem">
<img src="https://github.com/Aleshka5/Aviahack-Gym-Bros/blob/normal-gpu-calculating/gifs/luna_animation.gif" width="255" height="255" alt="lorem">
<img src="https://github.com/Aleshka5/Aviahack-Gym-Bros/blob/normal-gpu-calculating/gifs/agrad_animation.gif" width="255" height="255" alt="lorem"></p>


# Быстрое решение (CatBoost)
# Новые признаки
# Результаты
# Я не Torch
## Полносвязные модели
## Графовые модели
# Общие результаты 
# Заключение
# Ссылки
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

Чтобы запустить программу необходимо положить переданные модели на свои места согласно вышеуказанной структуре. После чего вы можете использовать наш класс ``` PressureModel() ``` как сторонний модуль. Основные функции которого:
``` python
pipeline()
```
Создать таблицу данных исходя из пути к файлу.
``` python 
predict()
```
Выдать предикт на таблицу данных.
``` python
__call__()
```
Вызывает пайплайн и предикт последовательно, чем обеспечивает удобное использование нашего сервиса из IDE Python.

Пример на python:
# Инициализация объекта класса
``` python
model = PressureModel()
```
# Получение предсказания в формате np.array()
``` python
result = model('ваш_путь_до_конкретного_файла_с_моделированием_давления/data_folder/princess_luna/0.3M/150')
```
