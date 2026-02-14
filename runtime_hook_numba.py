import os
import sys


def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        #jupyter lol
        return os.path.abspath(os.getcwd())
		
		
		
TOOL_DIR = get_app_dir()		
os.environ['NUMBA_CACHE_DIR'] = os.path.join(TOOL_DIR, '__numba_cache__')

from numba.core import config
import numba.core.caching as caching

class PyInstallerCacheLocator(caching.UserProvidedCacheLocator):
    @classmethod
    def from_function(cls, py_func, py_file):
        if not config.CACHE_DIR:
            return None
        
        # THE FIX: Allow if the file exists OR if we are running as a frozen PyInstaller app
        if not (os.path.exists(py_file) or getattr(sys, 'frozen', False)):
            return None
            
        self = cls(py_func, py_file)
        try:
            self.ensure_cache_path()
        except OSError:
            return None
            
        return self

# 3. Force Numba to use ONLY our new custom locator
caching.CacheImpl._locator_classes = [PyInstallerCacheLocator]

from numba import prange, njit, types
from numba.typed import Dict
from typing import List, Optional, Literal, Union
from numba.typed import List
from numba.core import event
import numba



print("Numba runtime hook executed, cache dir set to:", os.environ['NUMBA_CACHE_DIR'])