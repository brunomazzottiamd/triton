from collections.optional import Optional
from os.path import dirname, join, exists
from os.os import mkdir
from pathlib.path import cwd
from python import Python, PythonObject


alias DEFAULT_RNG_SEED: Int = 20250730


def numpy() -> PythonObject:
    return Python.import_module("numpy")


def gen_tensor(size: Int, rng_seed: Optional[Int] = DEFAULT_RNG_SEED) -> PythonObject:
    np = numpy()
    if rng_seed:
        np.random.seed(rng_seed.value())
    return np.random.randn(size).astype(np.float16)


def gen_tensor(shape: Tuple[Int, Int], rng_seed: Optional[Int] = DEFAULT_RNG_SEED) -> PythonObject:
    np = numpy()
    if rng_seed:
        np.random.seed(rng_seed.value())
    return np.random.randn(shape[0], shape[1]).astype(np.float16)


def tensor_equal(x: PythonObject, y: PythonObject) -> Bool:
    return Bool(numpy().array_equal(x, y))


def tensors_dir() -> String:
    tensors_dir = join(String(cwd()), "tensors")
    if not exists(tensors_dir):
        mkdir(tensors_dir)
    return tensors_dir


def tensor_file(tensor_name: String) -> String:
    return join(tensors_dir(), tensor_name + ".npz")


def save_tensor(tensor_name: String, x: PythonObject) -> None:
    numpy().savez_compressed(tensor_file(tensor_name), x)


def load_tensor(tensor_name: String) -> Optional[PythonObject]:
    f = tensor_file(tensor_name)
    if not exists(f):
        return None
    return numpy().load(f)["arr_0"]
