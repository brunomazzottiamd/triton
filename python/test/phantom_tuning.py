import torch
import triton
import triton.language as tl
from triton import Config


def get_hip_default_configs():
    configs=[
        Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16, "GROUP_M": 8, "SPLIT_K": 1, 'waves_per_eu': 2},
            num_warps=4,
            num_stages=2
        ),
        Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 16, "GROUP_M": 8, "SPLIT_K": 1, 'waves_per_eu': 2},
            num_warps=8,
            num_stages=2
        ),
    ]
    return configs


@triton.autotune(
    configs = get_hip_default_configs(),
    key=["M", "N", "K"],
    prune_configs_by= {}
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def triton_mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    dot_out_dtype: tl.constexpr,  #
    allow_tf32: tl.constexpr,  #
    fp8_fast_accum: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,  #
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    # rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    # rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] and (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        # tl.store(C, acc, mask=mask)
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def triton_mm(a, b, dot_out_dtype=None, allow_tf32=True, fp8_fast_accum=True):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = a.dtype
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32
    ab_dtype = True
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    print(f"a_shape = {a.shape}, b_shape = {b.shape}, c_shape = {c.shape}")

    triton_mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        dot_out_dtype=dot_out_dtype,  #
        allow_tf32=allow_tf32,  #
        fp8_fast_accum=fp8_fast_accum,  #
        # GROUP_M=8,
        AB_DTYPE=ab_dtype,
    )
    return c


def rand_strided(
    size,
    stride,
    dtype,
    device = "cpu",
    extra_size: int = 0,
):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(size, stride))
        + 1
        + extra_size
    )
    if dtype.is_floating_point:
        if dtype.itemsize == 1:
            """
            normal distribution kernel is not implemented for fp8..
            Workaround that by creating a fp16 tensor and then cast.
            """
            buffer = torch.randn(needed_size, dtype=torch.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)


def create_tensor(size, stride, device, dtype):
    if (
        stride is not None
        and len(stride) >= len(size)
        and all(isinstance(x, int) for x in stride)
    ):
        tensor = rand_strided(
            size=size,
            stride=stride,
            device=device,
            dtype=dtype,
        )
    else:
        tensor = torch.rand(
            size,
            device=device,
            dtype=dtype,
        )
    return tensor


if __name__ == "__main__":
    # Running benchmark for operator 8 / 50: (mm, c10::BFloat16, [[393216, 641], [641, 6514], [393216, 6514]], [[641, 1], [6514, 1], [6514, 1]])
    dtype = torch.bfloat16
    input_shape = [393216, 641]
    input_stride = [641, 1]
    mat2_shape = [641, 6514]
    mat2_stride = [6514, 1]
    gpu_device = torch.device("cuda")
    input_tensor = create_tensor(
        size=tuple(input_shape),
        stride=input_stride,
        device=gpu_device,
        dtype=dtype,
    )
    mat2_tensor = create_tensor(
        size=tuple(mat2_shape),
        stride=mat2_stride,
        device=gpu_device,
        dtype=dtype,
    )
    out_tensor = triton_mm(input_tensor, mat2_tensor)
    print(out_tensor)
