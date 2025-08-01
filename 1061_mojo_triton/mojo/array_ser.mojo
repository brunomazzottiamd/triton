from np_tensor import np_tensor


def main():
    npt = np_tensor()

    # Generate Mojo array.
    array_size = 10
    mojo_array = npt.gen_tensor(array_size)
    print("Generated Mojo array:")
    print(String(mojo_array))

    # Save Mojo array.
    npt.save_tensor("mojo_array", mojo_array)

    # Load Triton array and compare with Mojo array.
    triton_array = npt.load_tensor("triton_array")
    if triton_array is not None:
        print("Loaded Triton array:")
        print(String(triton_array))
        print(
            "Mojo array matches Triton array?",
            "Yes" if npt.tensor_equal(mojo_array, triton_array) else "No",
        )
    else:
        print("Couldn't find Triton array file.")
