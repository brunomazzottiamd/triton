import itertools
from math import e
from operator import itemgetter
from typing import Callable, Optional

import pytest
import torch
from scipy.stats import norm
from tabulate import tabulate
from torch import Tensor

import triton
import triton.language as tl

SUPPORTED_DTYPES: list[torch.dtype] = [torch.float16, torch.float32]
SUPPORTED_DTYPES_STR: list[str] = [
    str(dtype).replace("torch.", "").replace("float", "fp") for dtype in SUPPORTED_DTYPES
]


def dtype_to_str(dtype: torch.dtype) -> str:
    assert dtype in SUPPORTED_DTYPES
    return {dtype: dtype_str for dtype, dtype_str in zip(SUPPORTED_DTYPES, SUPPORTED_DTYPES_STR)}[dtype]


def str_to_dtype(dtype_str: str) -> torch.dtype:
    assert dtype_str in SUPPORTED_DTYPES_STR
    return {dtype_str: dtype for dtype_str, dtype in zip(SUPPORTED_DTYPES_STR, SUPPORTED_DTYPES)}[dtype_str]


def number_range(a: float, b: float, dtype: torch.dtype = torch.float32, step: Optional[float] = None) -> Tensor:
    assert a < b
    if not step:
        step = torch.finfo(dtype).eps
    assert step > 0
    n: int = int((b - a) / step)
    assert n >= 2
    return a + torch.arange(0, n, dtype=dtype, device="cuda") * ((b - a) / (n - 1))


def minus_half_plus_half_range(dtype: torch.dtype = torch.float32) -> Tensor:
    return number_range(-0.5, 0.5, dtype=dtype)


def minus_one_plus_one_range(dtype: torch.dtype = torch.float32) -> Tensor:
    return number_range(-1, 1, dtype=dtype)


def std_norm_range(p: float, dtype: torch.dtype = torch.float32, step: Optional[float] = None) -> Tensor:
    assert 0 < p < 1
    z_score: float = norm.ppf(1 - (1 - p) / 2)
    assert z_score > 0
    return number_range(-z_score, z_score, dtype=dtype, step=step)


def std_norm_nine_nines_range(dtype: torch.dtype = torch.float32) -> Tensor:
    return std_norm_range(0.999999999, dtype=dtype)


INPUT_GENERATORS: dict[str, Callable[[torch.dtype], Tensor]] = {
    "[-0.5, 0.5]": lambda dtype: minus_half_plus_half_range(dtype=dtype),
    "[-1, 1]": lambda dtype: minus_one_plus_one_range(dtype=dtype),
    "N(0, 1)": lambda dtype: std_norm_nine_nines_range(dtype=dtype),
}


# Subtracting the maximum value max(x) from x when computing softmax helps to improve numerical
# stability.
def normalize_for_softmax(x: Tensor) -> Tensor:
    return x - x.max()


def taylor_1st_at_zero(x: Tensor) -> Tensor:
    return 1 + x


def taylor_2nd_at_zero(x: Tensor) -> Tensor:
    return 1 + x + 0.5 * x * x


# -0.5 is z's mid-point when x ∈ [-0.5, 0.5].
def taylor_1st_at_minus_half(x: Tensor) -> Tensor:
    return 0.606531 + 0.606531 * (x + 0.5)


# -0.5 is z's mid-point when x ∈ [-0.5, 0.5].
def taylor_2nd_at_minus_half(x: Tensor) -> Tensor:
    x_plus_half: Tensor = x + 0.5
    return 0.606531 + 0.606531 * x_plus_half + 0.303265 * x_plus_half * x_plus_half


# -1 is z's mid-point when x ∈ [-1, 1].
def taylor_1st_at_minus_one(x: Tensor) -> Tensor:
    one_over_e: float = 1 / e
    return one_over_e + one_over_e * (x + 1)


# -1 is z's mid-point when x ∈ [-1, 1].
def taylor_2nd_at_minus_one(x: Tensor) -> Tensor:
    one_over_e: float = 1 / e
    x_plus_one: Tensor = x + 1
    return one_over_e + one_over_e * x_plus_one + (2 * one_over_e) * x_plus_one * x_plus_one


def cont_frac_1st(x: Tensor) -> Tensor:
    return 1 + x / (1 - 0.5 * x)


def cont_frac_2nd(x: Tensor) -> Tensor:
    return 1 + x / (1 - x / (2 + 1 / 3 * x))


def pade_1st(x: Tensor) -> Tensor:
    # y = (1 + x/2) / (1 - x/2)
    half_x: Tensor = 0.5 * x
    return (1 + half_x) / (1 - half_x)


def pade_2nd(x: Tensor) -> Tensor:
    # y = (1 + x/2 + x**2/12) / (1 - x/2 + x**2/12)
    # y = ((1 + x**2/12) + x/2) / ((1 + x**2/12) - x/2)
    # y = (a + b) / (a - b)
    a: Tensor = 1 + (1 / 12) * x * x
    b: Tensor = 0.5 * x
    return (a + b) / (a - b)


def pade_3rd(x: Tensor) -> Tensor:
    # y = (120 + 60 * x + 12 * x**2 + x**3) / (120 - 60 * x + 12 * x**2 - x**3)
    # y = ((120 + 12 * x**2) + (60 * x + x**3)) / ((120 + 12 * x**2) - (60 * x + x**3))
    # y = (a + b) / (a - b)
    x2: Tensor = x * x
    x3: Tensor = x * x2
    a: Tensor = 120 + 12 * x2
    b: Tensor = 60 * x + x3
    return (a + b) / (a - b)


def pade_4th(x: Tensor) -> Tensor:
    # y = (30240 + 15120 * x + 3360 * x**2 + 420 * x**3 + 30 * x**4) / (30240 - 15120 * x + 3360 * x**2 - 420 * x**3 + 30 * x**4)
    # y = ((30240 + 3360 * x**2 + 30 * x**4) + (15120 * x + 420 * x**3))  / ((30240 + 3360 * x**2 + 30 * x**4) - (15120 * x + 420 * x**3))
    # y = (a + b) / (a - b)
    x2: Tensor = x * x
    x3: Tensor = x * x2
    x4: Tensor = x2 * x2
    a: Tensor = 30240 + 3360 * x2 + 30 * x4
    b: Tensor = 15120 * x + 420 * x3
    return (a + b) / (a - b)


def range_reduction(x: Tensor, exp_fn: Callable[[Tensor], Tensor]) -> Tensor:
    log2: float = 0.6931471805599453  # log(2)
    one_over_log2: float = 1.4426950408889634  # 1 / log2

    n: Tensor
    # `torch.round` rounds to the nearest integer. Since `(1 / log2) * x` is never positive then we
    # can round to the nearest integer by subtracting 0.5 and truncating to integer. I couldn't find
    # `round` either in Triton documentation, it's going to be easier to port to Triton this way.
    n = (one_over_log2 * x - 0.5).to(torch.int32)
    # n can be computed more generally as follows:
    #     n = ((1 / log2) * x).round().to(torch.int32)

    r: Tensor = x - n * log2

    two_n: Tensor
    # Due to the working ranges of x and softmax normalization, i.e. z = x - max(x), n is never
    # positive. When x ∈ [-0.5, 0.5] then z ∈ [-1, 0]. When x ∈ [-1, 1] then z ∈ [-2, 0]. When
    # x ∈ N(0, 1) then z ∈ (-13, 0]. In practice, n will be something ∈ [-18, 0]. So we can compute
    # 2**n assuming it's never positive.
    two_n = 1 / (1 << -n)
    # 2**n can be computed as follows to cover both negative and positive values:
    #     two_n = torch.where(n >= 0, 1 << n, 1 / (1 << -n))

    return two_n * exp_fn(r)


EXP_APPROXIMATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "1st Taylor at 0": taylor_1st_at_zero,
    "2nd Taylor at 0": taylor_2nd_at_zero,
    "1st Taylor at -0.5": taylor_1st_at_minus_half,
    "2nd Taylor at -0.5": taylor_2nd_at_minus_half,
    "1st Taylor at -1": taylor_1st_at_minus_one,
    "2nd Taylor at -1": taylor_2nd_at_minus_one,
    "1st Continued Fraction": cont_frac_1st,
    "2nd Continued Fraction": cont_frac_2nd,
    "1st Padé": pade_1st,
    "2nd Padé": pade_2nd,
    "3rd Padé": pade_3rd,
    "4th Padé": pade_4th,
}

EXP_APPROXIMATIONS.update({
    f"Range Reduction + {approx}": lambda x, approx_fn=approx_fn: range_reduction(x, approx_fn)  # type: ignore
    for approx, approx_fn in EXP_APPROXIMATIONS.items()
})


def max_abs_err(y_true: Tensor, y_approx: Tensor) -> float:
    return (y_true - y_approx).abs().max().item()


def evaluate_exp_approximations(dtype: torch.dtype = torch.float32) -> None:
    print(f"\nEvaluating approximations for exp(x - max(x)) using {dtype_to_str(dtype)}...")
    eps: float = torch.finfo(dtype).eps

    for input_gen, input_gen_fn in INPUT_GENERATORS.items():
        print(f"\nGenerating z = x - max(x) where x ∈ {input_gen} with step {eps:.2e}...")
        z: Tensor = normalize_for_softmax(input_gen_fn(dtype))
        print(f"z has {z.shape[0]} elements ∈ [{z.min():.6f}, {z.max():.6f}].")

        print("Computing reference y = exp(z)...")
        y_true: Tensor = z.exp()

        table: list[list[str | float]] = []

        min_error: float = float("inf")
        best_approx: str = "None"

        for approx, approx_fn in EXP_APPROXIMATIONS.items():
            y_approx: Tensor = approx_fn(z)
            approx_error: float = max_abs_err(y_true, y_approx)

            table.append([approx, approx_error])

            if approx_error < min_error:
                min_error = approx_error
                best_approx = approx

        table = sorted(table, key=itemgetter(1))
        print(tabulate(table, headers=["Approximation", "Max. Abs. Error ↓"], floatfmt=("", ".2e")))
        print(f"For x ∈ {input_gen}, the best approximation is {best_approx}.")


def get_autotune_config() -> list[triton.Config]:
    waves_per_eu_range: list[int] = [1, 2, 4]
    num_warps_range: list[int] = [4, 8, 16]
    return [
        triton.Config({"waves_per_eu": waves_per_eu}, num_warps=num_warps, num_stages=1)
        for waves_per_eu, num_warps in itertools.product(waves_per_eu_range, num_warps_range)
    ]


@triton.jit
def exp_approx(z):
    # Range reduction from z to r:
    #     Constants:
    #           log(2) = 0.6931471805599453
    #       1 / log(2) = 1.4426950408889634
    n = (1.4426950408889634 * z - 0.5).to(tl.int32)
    r = z - n * 0.6931471805599453
    # Compute exp(r) using 3rd order Padé approximation:
    r2 = r * r
    r3 = r * r2
    a = 120 + 12 * r2
    b = 60 * r + r3
    exp_r = tl.fdiv(a + b, a - b)
    # This also works for exp_r:
    #     exp_r = (a + b) / (a - b)
    # Reverse range reduction:
    two_n = 1 / (1 << -n)
    # This also works for two_n:
    #     (both operands of tl.fdiv must be floating point types)
    #     two_n = tl.fdiv(tl.full((BLOCK_SIZE, ), 1, tl.float32), (1 << -n).to(tl.float32))
    return two_n * exp_r


@triton.autotune(configs=get_autotune_config(), key=["n", "EXP_METHOD"])
@triton.heuristics({"EVEN_N": lambda args: args["n"] % args["BLOCK_SIZE"] == 0})
@triton.jit
def triton_exp_kernel(z_ptr, y_ptr, n: int,  #
                      BLOCK_SIZE: tl.constexpr, EXP_METHOD: tl.constexpr,  #
                      EVEN_N: tl.constexpr):
    # Compute pointer offsets:
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load block:
    if not EVEN_N:
        mask = offsets < n
        z = tl.load(z_ptr + offsets, mask=mask, other=0)
    else:
        z = tl.load(z_ptr + offsets)

    # Always cast input to float32:
    z = z.to(tl.float32)

    if EXP_METHOD == "triton_exp":
        y = tl.exp(z)

    if EXP_METHOD == "triton_exp2":
        # exp(x) = exp2(x / log(2)) = exp2((1 / log(2) * x)
        # 1 / log(2) = 1.4426950408889634
        y = tl.exp2(1.4426950408889634 * z)

    if EXP_METHOD == "triton_exp_approx":
        y = exp_approx(z)

    # Cast back to output data type:
    y = y.to(y_ptr.type.element_ty)

    # Store block:
    if not EVEN_N:
        tl.store(y_ptr + offsets, y, mask=mask)
    else:
        tl.store(y_ptr + offsets, y)


def triton_exp(exp_method: str, z: Tensor) -> Tensor:
    assert exp_method in ["triton_exp", "triton_exp2", "triton_exp_approx"]

    n: int = z.numel()

    max_fused_size: int = 65536 // z.element_size()
    block_size: int = min(triton.next_power_of_2(n), max_fused_size)

    grid: tuple[int] = (triton.cdiv(n, block_size), )

    y: Tensor = torch.empty_like(z)

    triton_exp_kernel[grid](z, y, n, block_size, exp_method)

    return y


def exp(exp_method: str, z: Tensor) -> Tensor:
    assert exp_method in ["torch", "triton_exp", "triton_exp2", "triton_exp_approx"]
    return z.exp() if exp_method == "torch" else triton_exp(exp_method, z)


@pytest.mark.parametrize("input_gen_str", ["minus_half_plus_half", "minus_one_plus_one", "std_norm"])
@pytest.mark.parametrize("dtype_str", SUPPORTED_DTYPES_STR)
def test_exp(dtype_str: str, input_gen_str: str) -> None:
    input_gen: Callable[[torch.dtype], Tensor] = {
        "minus_half_plus_half": minus_half_plus_half_range,
        "minus_one_plus_one": minus_one_plus_one_range,
        "std_norm": std_norm_nine_nines_range,
    }[input_gen_str]
    z: Tensor = normalize_for_softmax(input_gen(str_to_dtype(dtype_str)))

    y_torch: Tensor = exp("torch", z)
    y_triton_exp: Tensor = exp("triton_exp", z)
    y_triton_exp2: Tensor = exp("triton_exp2", z)
    y_triton_exp_approx: Tensor = exp("triton_exp_approx", z)

    eps: float = 1.2e-7
    assert max_abs_err(y_torch, y_triton_exp) < eps, "Triton exp doesn't match PyTorch exp."
    assert max_abs_err(y_torch, y_triton_exp2) < eps, "Triton exp2 doesn't match PyTorch exp."
    assert max_abs_err(y_torch, y_triton_exp_approx) < eps, "Triton exp approximation doesn't match PyTorch exp."


def benchmark_exp(dtype: torch.dtype = torch.float32) -> None:
    perf_unit: str = "GiB/s"
    line_vals: list[str] = ["torch", "triton_exp", "triton_exp2", "triton_exp_approx"]
    line_names: list[str] = [f"{x.replace('_', ' ').title()} ({perf_unit})" for x in line_vals]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["size"],
            x_vals=[2**i for i in range(8, 22, 1)],
            xlabel="Size",
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            ylabel=perf_unit,
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("purple", "-")],
            plot_name=f"{dtype_to_str(dtype)}_exp_performance",
            args={},
        ))
    def benchmark(provider: str, size: int):
        assert size > 0

        z: Tensor = normalize_for_softmax(torch.randn(size, dtype=dtype, device="cuda"))
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: exp(provider, z), quantiles=[0.5, 0.2, 0.8])

        if "triton" in provider:
            print(f"provider={provider} size={size} best_config={triton_exp_kernel.best_config}")

        # Convert milliseconds to GiB/s.
        def gibps(ms: float) -> float:
            return (2 * z.numel() * z.element_size() * 2**-30) / (ms * 1e-3)

        return gibps(ms), gibps(max_ms), gibps(min_ms)

    print(f"\nBenchmarking Triton kernel using {dtype_to_str(dtype)}...")
    benchmark.run(print_data=True, show_plots=False)


def main() -> None:
    for dtype in SUPPORTED_DTYPES:
        evaluate_exp_approximations(dtype=dtype)
        benchmark_exp(dtype=dtype)


if __name__ == "__main__":
    main()
