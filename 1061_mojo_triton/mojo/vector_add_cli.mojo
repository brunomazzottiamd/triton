from python import Python, PythonObject
from sys import argv, exit

from python_import import import_module


@fieldwise_init
struct VectorAddArgs(Copyable, Movable):
    var n: List[Int]
    var runs: Int
    var save_out: Bool


def parse_args() -> VectorAddArgs:
    argv = argv()
    argc = len(argv)
    str_args = List[String]()
    for i in range(1, argc):
        str_args.append(argv[i])
    parsed_args = import_module("vector_add_cli").parse_args(Python.list(str_args))
    ns = List[Int]()
    for n in parsed_args.n:
        ns.append(Int(n))
    return VectorAddArgs(ns, Int(parsed_args.runs), Bool(parsed_args.save_out))
