from python import Python, PythonObject
from collections.optional import Optional


alias DEFAULT_RNG_SEED: Int = 20250730


def gen_tensor(size: Int, rng_seed: Optional[Int] = DEFAULT_RNG_SEED) -> PythonObject:
    np = Python.import_module("numpy")
    if rng_seed:
        np.random.seed(rng_seed.value())
    return np.random.randn(size).astype(np.float16)


def gen_tensor(shape: Tuple[Int, Int], rng_seed: Optional[Int] = DEFAULT_RNG_SEED) -> PythonObject:
    np = Python.import_module("numpy")
    if rng_seed:
        np.random.seed(rng_seed.value())
    return np.random.randn(shape[0], shape[1]).astype(np.float16)
