import os
import numpy as np


# Generate Triton array.
rng = np.random.default_rng(seed=20250730)
array_size = 10
triton_array = rng.standard_normal(size=array_size, dtype=np.float32)
triton_array = triton_array.astype(np.float16)

# Get directory of serialized arrays.
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
arrays_dir = os.path.dirname(script_dir)

# Save Triton array.
triton_array_file = os.path.join(arrays_dir, "triton_array.npz")
np.savez(triton_array_file, triton_array=triton_array)

# Load Mojo array and compare with Triton array.
mojo_array_file = os.path.join(arrays_dir, "mojo_array.npz")
if os.path.exists(mojo_array_file):
    mojo_array = np.load(mojo_array_file)["mojo_array"]
    equal = np.array_equal(triton_array, mojo_array)
    print("Triton array matches Mojo array?", "Yes" if equal else "No")
else:
    print("Couldn't find Mojo array file.")
