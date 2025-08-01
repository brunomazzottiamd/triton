from os.path import join
from pathlib.path import cwd
from python import Python, PythonObject


def np_tensor() -> PythonObject:
    Python.add_to_path(join(String(cwd()), "python"))
    return Python.import_module("np_tensor")
