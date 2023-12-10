import os
import glob

def get_model_path(geometry_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = glob.glob(os.path.join(current_dir, "*.cbm"))
    for model_file in model_files:
        if geometry_type in model_file:
            return model_file

    model_path = model_files[0]
    return model_path
