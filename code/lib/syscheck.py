'''
Based on instructions and code from https://realpython.com/python313-free-threading-jit/

Bartosz Zaczyński--"Python 3.13 Preview: Free Threading and a JIT Compiler" Real Python , Sep 18 2024.
'''

# This turned out to not be necessary. Keeping it as it may be helpful later on.

import sys 
import sysconfig 

def syscheck():
    if not sys.version_info >= (3, 13):
        raise RuntimeError(f"Python 3.13t or newer (with free-threading enabled) required.\nCurrent Python version:{sys.version_info}")
    if not sysconfig.get_config_var("Py_GIL_DISABLED") == 1 or sys._is_gil_enabled():
        raise RuntimeError("Free-threading not supported by current Python interpreter. Config var Py_GIL_DISABLED != 1")
    if sys._is_gil_enabled(): 
        raise RuntimeError("Free-threading supported, but not enabled by current Python interpreter. sys._is_gil_enabled() returned True.")

