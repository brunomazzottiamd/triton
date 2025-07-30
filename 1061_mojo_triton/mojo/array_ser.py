import os
import numpy as np


# Generate Mojo array.
rng = np.random.default_rng(seed=20250730)
array_size = 10
mojo_array = rng.standard_normal(size=array_size, dtype=np.float32)
mojo_array = mojo_array.astype(np.float16)

# Get directory of serialized arrays.
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
arrays_dir = os.path.dirname(script_dir)

# Save Mojo array.
mojo_array_file = os.path.join(arrays_dir, "mojo_array.npz")
np.savez(mojo_array_file, mojo_array=mojo_array)

# Load Triton array and compare with Mojo array.
triton_array_file = os.path.join(arrays_dir, "triton_array.npz")
if os.path.exists(triton_array_file):
    triton_array = np.load(triton_array_file)["triton_array"]
    equal = np.array_equal(mojo_array, triton_array)
    print("Mojo array matches Triton array?", "Yes" if equal else "No")
else:
    print("Couldn't find Triton array file.")
