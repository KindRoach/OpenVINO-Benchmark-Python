#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

method=${1:-opencv.sync_decode}
run_id=$(date "+%Y-%m-%d-%H-%M-%S")
n_core=$(nproc --all)
n_process_list=()
for i in $(seq 1 "$n_core"); do
    n=$((n_core / i * i))
    [ $n == "$n_core" ] && n_process_list+=("$i")
done

for n in "${n_process_list[@]}"; do
    log_dir=./outputs/benchmark/$method/$run_id/$n
    mkdir -p "$log_dir"
    n_core_per_process=$((n_core / n))
    for ((i = 0; i < n; i++)); do
        core_start=$((i * n_core_per_process))
        core_end=$((core_start + n_core_per_process - 1))
        numactl -C "$core_start-$core_end" python -m "$method" >"$log_dir/$i.out" &
    done
    wait
    echo "$log_dir done"
done
