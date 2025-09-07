import os
import inspect

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