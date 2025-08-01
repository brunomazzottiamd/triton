from np_tensor import gen_tensor, save_tensor, load_tensor, tensor_equal


def main() -> None:
    # Generate Triton array.
    array_size = 10
    triton_array = gen_tensor(array_size)
    print("Generated Triton array:")
    print(triton_array)

    # Save Triton array.
    save_tensor("triton_array", triton_array)

    # Load Mojo array and compare with Triton array.
    mojo_array = load_tensor("mojo_array")
    if mojo_array is not None:
        print("Loaded Mojo array:")
        print(mojo_array)
        print(
            "Triton array matches Mojo array?",
            "Yes" if tensor_equal(triton_array, mojo_array) else "No",
        )
    else:
        print("Couldn't find Mojo array file.")


if __name__ == "__main__":
    main()
