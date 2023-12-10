import os
import glob

def get_model_path(geometry_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = glob.glob(os.path.join(current_dir, "*.tph"))
    for model_file in model_files:
        if geometry_type in model_file:
            return model_file

    # Модель не найдена
    return None
