# Выберите базовый образ
from pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Базовые настройки параметров
ENV SHELL=/bin/bash
EXPOSE 8888

# Установка нужных утилит в ОС
RUN apt-get update && apt-get install -y \ 
    sudo \
    apt-utils \
    vim \
    git 

RUN pip install \
    jupyterlab \
    jupyter -U
    

# Установка библиотек питона
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torch-geometric
RUN pip install -q git+https://github.com/snap-stanford/deepsnap.git
RUN pip install pandas
RUN pip install matplotlib
RUN pip install h5py
RUN pip install tensorflow
RUN pip install numba
RUN pip install seaborn
RUN pip install torchmetrics
RUN pip install ipympl
RUN pip install git+https://github.com/niasw/openfoamparser.git


# Точка входа - юпитер ноутбук для удобства работы с файлами
WORKDIR /home/jovyan

ENTRYPOINT jupyter lab \
    --notebook-dir=/home/jovyan \
    --ip=0.0.0.0 \
    --no-browser \
    --allow-root \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --FileContentsManager.delete_to_trash=True
