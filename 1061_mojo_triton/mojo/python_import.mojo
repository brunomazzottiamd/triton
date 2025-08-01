from os.path import join
from pathlib.path import cwd
from python import Python, PythonObject


def import_module(module_name: String) -> PythonObject:
    Python.add_to_path(join(String(cwd()), "python"))
    return Python.import_module(module_name)
