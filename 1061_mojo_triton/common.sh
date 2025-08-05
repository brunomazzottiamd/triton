### Common variables

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_DIR="${SCRIPT_DIR}/python"
MOJO_DIR="${SCRIPT_DIR}/mojo"
TENSORS_DIR="${SCRIPT_DIR}/tensors"
PROFILING_DIR="${SCRIPT_DIR}/profiling"

### Generic helper functions

remove() {
    rm --recursive --force "${@}"
}

create_dir() {
    mkdir --parents "${1}"
}

show_progress() {
    local bar_size=40
    local bar_char_done='#'
    local bar_char_todo='-'
    local bar_percentage_scale=2
    local current="${1}"
    local total="${2}"
    local percent
    percent=$(bc <<< "scale=${bar_percentage_scale}; 100 * ${current} / ${total}")
    local done_
    done_=$(bc <<< "scale=0; ${bar_size} * ${percent} / 100")
    local todo
    todo=$(bc <<< "scale=0; ${bar_size} - ${done_}")
    local done_sub_bar
    done_sub_bar=$(printf "%${done_}s" | tr ' ' "${bar_char_done}")
    local todo_sub_bar
    todo_sub_bar=$(printf "%${todo}s" | tr ' ' "${bar_char_todo}")
    local bar
    bar=$(printf "[${done_sub_bar}${todo_sub_bar}] [%3d /%3d] [${percent}%%]" "${current}" "${total}")
    echo -ne "\r${bar}"
}

### Helper functions for Mojo + Triton project

clean_tensors() {
    local tensor_pattern="${1}"
    remove "${TENSORS_DIR}/"*"${tensor_pattern}"*
}

run_python() {
    local python_program="${1}"
    shift
    python "${PYTHON_DIR}/${python_program}.py" "${@}"
}

run_mojo() {
    local mojo_program="${1}"
    shift
    pixi run --manifest-path="${MOJO_DIR}/pixi.toml" \
        mojo run "${MOJO_DIR}/${mojo_program}.mojo" "${@}"
}

run_test() {
    local test="${1}"
    pytest --no-header --no-summary "${PYTHON_DIR}/test_${test}.py"
}

### Kernel profiling functions

profile_kernel() {
    local num_executions="${1}"
    shift
    local regex="${1}"
    shift
    local output_csv_file="${1}"
    shift
    echo 'kernel_name,duration_ns' > "${output_csv_file}"
    # echo "Running [ ${num_executions} ] rocprof executions:"
    for ((i = 1; i <= num_executions; ++i)); do
        show_progress "${i}" "${num_executions}"
        rocprof --stats "${@}" &> /dev/null
        grep "${regex}" results.stats.csv \
            | csvcut --columns=1,3 \
            >> "${output_csv_file}"
        remove results.*
    done
    echo
}

compute_kernel_stats() {
    local input_csv_file="${1}"
    local output_csv_file="${2}"
    # echo 'Statistics:'
    python << EOF
import pandas as pd

df = pd.read_csv("${input_csv_file}")
if df.empty:
    print("WARNING: Profiling data file is empty.")
    import sys
    sys.exit()

df["duration_us"] = df["duration_ns"] / 10**3
df.drop(columns=["duration_ns"], inplace=True)

def trimmed_stats(x, trim_percent=0.2):
    x = x.sort_values().iloc[
        int(len(x) * trim_percent) : int(len(x) * (1 - trim_percent))
    ]
    return pd.Series({"mean_us": x.mean().round(2), "std_us": x.std(ddof=0).round(2)})

df = (
    df.groupby("kernel_name")["duration_us"]
    .apply(trimmed_stats)
    .unstack()
    .reset_index()
)
df.to_csv("${output_csv_file}", index=False)
# print(df.to_markdown(index=False))
EOF
}

profile() {
    local kernel_regex="${1}"
    shift
    local output_dir="${PROFILING_DIR}/${1}"
    shift
    local num_executions=25
    local prof_data_file="${output_dir}/prof_data.csv"
    local prof_stats_file="${output_dir}/prof_stats.csv"
    create_dir "${output_dir}"
    profile_kernel "${num_executions}" "${kernel_regex}" "${prof_data_file}" "${@}"
    compute_kernel_stats "${prof_data_file}" "${prof_stats_file}"
}

profile_python() {
    local python_program="${1}"
    shift
    local kernel_regex="${1}"
    shift
    local output_dir="${1}/python"
    shift
    profile "${kernel_regex}" "${output_dir}" \
        python "${PYTHON_DIR}/${python_program}.py" "${@}"
}

profile_mojo() {
    local mojo_program="${1}"
    shift
    local kernel_regex="${1}"
    shift
    local output_dir="${1}/mojo"
    shift
    profile "${kernel_regex}" "${output_dir}" \
        pixi run --manifest-path="${MOJO_DIR}/pixi.toml" \
        mojo run "${MOJO_DIR}/${mojo_program}.mojo" "${@}"
}
