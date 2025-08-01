from python import PythonObject

from python_import import import_module


def np_tensor() -> PythonObject:
    return import_module("np_tensor")
