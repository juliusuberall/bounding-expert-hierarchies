import os
import inspect
import shutil
from datetime import datetime

def setup_dirs(module):
    '''
    Create all directories from ./registry module to ensure all neccessary folder are in place.
    '''
    for _ , registry in inspect.getmembers(module):
        if isinstance(registry, dict):
            if len(registry.values()) > 4 : continue
            for dir in registry.values():
                if dir == None : continue
                if dir.count(".") >= 2 : continue
                os.makedirs(dir, exist_ok=True)
    pass

def move_results(folder : str, prefix : str):
    '''
    Move all results into folder to group mass experiments by data name prefix.
    '''
    # Create folder name with timestamp if prefix-folder already exists
    x_folder = os.path.join(folder, prefix)
    if os.path.exists(x_folder):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        x_folder = os.path.join(folder, f"{prefix}_{timestamp}")

    os.makedirs(x_folder, exist_ok=True)

    # Move all files starting with prefix
    for filename in os.listdir(folder):
        if filename.startswith(prefix):
            src = os.path.join(folder, filename)
            dst = os.path.join(x_folder, filename)
            if os.path.isfile(src):
                shutil.move(src, dst)
                print(f"Moved: {filename}")