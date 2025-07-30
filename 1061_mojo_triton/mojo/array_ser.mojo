from os.path import dirname, join, exists
from pathlib.path import cwd
from python import Python

from common import gen_tensor


def main():
    # Import Python's NumPy.
    np = Python.import_module("numpy")

    # Generate Mojo array.
    array_size = 10
    mojo_array = gen_tensor(array_size)

    # Get directory of serialized arrays.
    arrays_dir = String(cwd())

    # Save Mojo array.
    mojo_array_file = join(arrays_dir, "mojo_array.npz")
    np.savez(mojo_array_file, mojo_array=mojo_array)

    # Load Triton array and compare with Mojo array.
    triton_array_file = join(arrays_dir, "triton_array.npz")
    if exists(triton_array_file):
        triton_array = np.load(triton_array_file)["triton_array"]
        equal = np.array_equal(mojo_array, triton_array)
        print("Mojo array matches Triton array?", "Yes" if equal else "No")
    else:
        print("Couldn't find Triton array file.")
