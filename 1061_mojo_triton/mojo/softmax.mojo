from gpu.host import DeviceContext
from python import Python, PythonObject
from sys import exit
from sys.info import has_accelerator

from python_interop import np_tensor, parse_softmax_args


def softmax(
    ctx: DeviceContext,
    x: PythonObject,
) -> PythonObject:
    # Create output NumPy array.
    shape = x.shape
    y = np_tensor().empty_tensor(shape)

    # Get host pointers from underlying NumPy arrays.
    x_host_ptr = x.ctypes.data.unsafe_get_as_pointer[DType.float16]()
    y_host_ptr = y.ctypes.data.unsafe_get_as_pointer[DType.float16]()

    # Create device buffers.
    buffer_size = Int(shape[0]) * Int(shape[1])
    x_device_buf = ctx.enqueue_create_buffer[DType.float16](buffer_size)
    y_device_buf = ctx.enqueue_create_buffer[DType.float16](buffer_size)

    # Copy from host to device.
    ctx.enqueue_copy(dst_buf=x_device_buf, src_ptr=x_host_ptr)

    # Invoke kernel.
    # TODO: Write Mojo kernel and launch it here!

    # Copy from device to host.
    ctx.enqueue_copy(dst_ptr=y_host_ptr, src_buf=y_device_buf)

    # Wait for all device operations to complete.
    ctx.synchronize()

    return y


def run_softmax(
    ctx: DeviceContext,
    shapes: List[Tuple[Int, Int]], runs: Int, save_tensors: Bool, verbose: Bool,
) -> None:
    npt = np_tensor()
    for shape in shapes:
        m, n = shape[0], shape[1]
        x = npt.gen_tensor(Python.tuple(m, n))
        if save_tensors:
            if verbose:
                print("Saving softmax input for shape=(" + String(m) + ", " + String(n) + ").")
            tensor_name = "mojo___softmax_x_" + String(m).rjust(5, "0") + "_" + String(n).rjust(5, "0")
            npt.save_tensor(tensor_name, x)
        if verbose:
            print("Running softmax for shape=(" + String(m) + ", " + String(n) + ") (1 / " + String(runs) + ").")
        y = softmax(ctx, x)
        for i in range(2, runs + 1):
            if verbose:
                print("Running softmax for shape=(" + String(m) + ", " + String(n) + ") (" + String(i) + " / " + String(runs) + ").")
            y = softmax(ctx, x)
        if save_tensors:
            if verbose:
                print("Saving softmax output for shape=(" + String(m) + ", " + String(n) + ").")
            tensor_name = "mojo___softmax_y_" + String(m).rjust(5, "0") + "_" + String(n).rjust(5, "0")
            npt.save_tensor(tensor_name, y)


def main():
    @parameter
    if not has_accelerator():
        print("No GPU detected.")
        exit(1)

    try:
        args = parse_softmax_args()
        ctx = DeviceContext()
        run_softmax(ctx, args.shape, args.runs, args.save_tensors, args.verbose)
    except:
        exit(1)
