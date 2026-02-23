#!/bin/bash
set -e

run() {
    dataset=$1
    end=$2
    mode=$3
    echo "Running $dataset $mode..."
    micromamba run -n timesfm311 python baselines_2of4.py --csv ETDataset/ETT-small/$dataset.csv --col OT --mode $mode --train_end $end > logs/${dataset}_${mode}.log 2>&1
}

mkdir -p logs

# ETTm1
run "ETTm1" 49152 "magnitude"
run "ETTm1" 49152 "wanda"
run "ETTm1" 49152 "sparsegpt"

# ETTm2
run "ETTm2" 49152 "magnitude"
run "ETTm2" 49152 "wanda"
run "ETTm2" 49152 "sparsegpt"

# ETTh1
run "ETTh1" 8640 "magnitude"
run "ETTh1" 8640 "wanda"
run "ETTh1" 8640 "sparsegpt"

# ETTh2
run "ETTh2" 8640 "magnitude"
run "ETTh2" 8640 "wanda"
run "ETTh2" 8640 "sparsegpt"

echo "All baselines completed."
