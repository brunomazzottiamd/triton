* `opt00_baseline-tl-constexpr.diff`:
  * Use `tl.constexpr` as much as possible.
  * Base commit: `630a1266a`
* `opt01_baseline-tune-gemm-group-m.diff`:
  * Use L2 grouping from `tune_gemm.py`.
  * Base commit: `630a1266a`
* `opt02_nonkdim16-tune-gemm-group-m.diff`:
  * Use L2 grouping from `tune_gemm.py`.
  * Base commit: `2a24b9cfc`
* `opt03_nonkdim16-tl-constexpr.diff`:
  * Use `tl.constexpr` as much as possible.
  * Base commit: `2a24b9cfc`
* `opt04_autotune-cache-wt.diff`:
  * Use cache write-through `cache_modifier=".wt"` in `tl.store`.
  * Base commit: `dd5cbc1b5`
* `opt05_autotune-tune-gemm-group-m.diff`:
  * Use L2 grouping from `tune_gemm.py`.
  * Base commit: `dd5cbc1b5`
