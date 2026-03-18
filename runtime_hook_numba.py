import os
import sys
import multiprocessing
import shutil
import tempfile

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        #jupyter lol
        return os.path.abspath(os.getcwd())
    
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x0001 | 0x8000)


multiprocessing.freeze_support()		
		
		
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

caching.CacheImpl._locator_classes = [PyInstallerCacheLocator]


#MO2 workaround
_real_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import customtkinter #surpressing error on init so MO2 people don't complain
sys.stderr.close()
sys.stderr = _real_stderr

assets_fonts = os.path.join(get_app_dir(), "_internal", "customtkinter",
                            "assets", "fonts")

original_load = customtkinter.FontManager.windows_load_font

@classmethod
def patched_load_font(cls, font_path, *args, **kwargs):
    if os.path.isfile(font_path):
        temp_font_dir = os.path.join(tempfile.gettempdir(), "ctk_fonts_cache")
        os.makedirs(temp_font_dir, exist_ok=True)
        dest = os.path.join(temp_font_dir, os.path.basename(font_path))
        shutil.copy2(font_path, dest)
        print(font_path)
        result = original_load.__func__(cls, dest, *args, **kwargs)	
        return result
    return original_load.__func__(cls, font_path, *args, **kwargs)
    
customtkinter.FontManager.windows_load_font = patched_load_font
customtkinter.FontManager.load_font(os.path.join(assets_fonts, "CustomTkinter_shapes_font.otf"))
customtkinter.FontManager.load_font(os.path.join(assets_fonts, "Roboto", "Roboto-Regular.ttf"))
customtkinter.FontManager.load_font(os.path.join(assets_fonts, "Roboto", "Roboto-Medium.ttf"))
customtkinter.DrawEngine.preferred_drawing_method = "font_shapes"