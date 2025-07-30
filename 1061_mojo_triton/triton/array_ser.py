from common import gen_tensor, save_tensor, load_tensor, tensor_equal


# Generate Triton array.
array_size = 10
triton_array = gen_tensor(array_size)

# Save Triton array.
save_tensor("triton_array", triton_array)

# Load Mojo array and compare with Triton array.
mojo_array = load_tensor("mojo_array")
if mojo_array is not None:
    print(
        "Triton array matches Mojo array?",
        "Yes" if tensor_equal(triton_array, mojo_array) else "No",
    )
else:
    print("Couldn't find Mojo array file.")
