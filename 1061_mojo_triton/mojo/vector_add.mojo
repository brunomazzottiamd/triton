from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from math import ceildiv
from python import PythonObject
from sys import exit
from sys.info import has_accelerator

from np_tensor import np_tensor
from vector_add_cli import parse_args


fn vector_add_kernel(
    x: UnsafePointer[Float16],
    y: UnsafePointer[Float16],
    z: UnsafePointer[Float16],
    n: Int,
) -> None:
    idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < n:
        z[idx] = x[idx] + y[idx]


def vector_add(
    ctx: DeviceContext,
    x: PythonObject, y: PythonObject,
) -> PythonObject:
    # Create output NumPy array.
    n = Int(x.size)
    z = np_tensor().empty_tensor(n)

    # Get host pointers from underlying NumPy arrays.
    x_host_ptr = x.ctypes.data.unsafe_get_as_pointer[DType.float16]()
    y_host_ptr = y.ctypes.data.unsafe_get_as_pointer[DType.float16]()
    z_host_ptr = z.ctypes.data.unsafe_get_as_pointer[DType.float16]()

    # Create device buffers.
    x_device_buf = ctx.enqueue_create_buffer[DType.float16](n)
    y_device_buf = ctx.enqueue_create_buffer[DType.float16](n)
    z_device_buf = ctx.enqueue_create_buffer[DType.float16](n)

    # Copy from host to device.
    ctx.enqueue_copy(dst_buf=x_device_buf, src_ptr=x_host_ptr)
    ctx.enqueue_copy(dst_buf=y_device_buf, src_ptr=y_host_ptr)

    # Invoke kernel.
    block_size = 1024
    num_blocks = ceildiv(n, block_size)
    ctx.enqueue_function[vector_add_kernel](
        x_device_buf,
        y_device_buf,
        z_device_buf,
        grid_dim=num_blocks,
        block_dim=block_size,
    )

    # Copy from device to host.
    ctx.enqueue_copy(dst_ptr=z_host_ptr, src_buf=z_device_buf)

    # Wait for all device operations to complete.
    ctx.synchronize()

    return z


def run_vector_add(
    ctx: DeviceContext,
    ns: List[Int], runs: Int, save_out: Bool,
) -> None:
    npt = np_tensor()
    for n in ns:
        x = npt.gen_tensor(n)
        y = npt.gen_tensor(n, rng_seed=None)
        z = vector_add(ctx, x, y)
        for _ in range(0, runs - 1):
            z = vector_add(ctx, x, y)
        if save_out:
            tensor_name = "mojo_vector_add_" + String(n).rjust(8, "0")
            npt.save_tensor(tensor_name, z)


def main():
    @parameter
    if not has_accelerator():
        print("No GPU detected.")
        exit(1)

    try:
        args = parse_args()
        ctx = DeviceContext()
        run_vector_add(ctx, args.n, args.runs, args.save_out)
    except:
        exit(1)
