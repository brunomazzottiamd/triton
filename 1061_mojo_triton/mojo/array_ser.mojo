from common import gen_tensor, save_tensor, load_tensor, tensor_equal


def main():
    # Generate Mojo array.
    array_size = 10
    mojo_array = gen_tensor(array_size)

    # Save Mojo array.
    save_tensor("mojo_array", mojo_array)

    # Load Triton array and compare with Mojo array.
    triton_array = load_tensor("triton_array")
    if triton_array is not None:
        print(
            "Mojo array matches Triton array?",
            "Yes" if tensor_equal(mojo_array, triton_array.value()) else "No",
        )
    else:
        print("Couldn't find Triton array file.")
