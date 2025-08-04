from os.path import join
from pathlib.path import cwd
from python import Python, PythonObject
from sys import argv, exit


def import_module(module_name: String) -> PythonObject:
    Python.add_to_path(join(String(cwd()), "python"))
    return Python.import_module(module_name)


def np_tensor() -> PythonObject:
    return import_module("np_tensor")


def parse_args(module_name: String) -> PythonObject:
    argv = argv()
    argc = len(argv)
    str_args = List[String]()
    for i in range(1, argc):
        str_args.append(argv[i])
    return import_module(module_name).parse_args(Python.list(str_args))


@fieldwise_init
struct VectorAddArgs(Copyable, Movable):
    var n: List[Int]
    var runs: Int
    var save_out: Bool
    var verbose: Bool


def parse_vector_add_args() -> VectorAddArgs:
    args = parse_args("vector_add_cli")
    ns = List[Int]()
    for n in args.n:
        ns.append(Int(n))
    return VectorAddArgs(
        ns, Int(args.runs), Bool(args.save_out), Bool(args.verbose)
    )
