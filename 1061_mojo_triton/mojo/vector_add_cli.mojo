from python import PythonObject

from python_import import import_module


def parse_args() -> PythonObject:
    return import_module("vector_add_cli").parse_args()
