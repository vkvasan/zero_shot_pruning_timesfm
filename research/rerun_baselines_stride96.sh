#!/bin/bash
# Re-run ALL baselines (SparseGPT + Magnitude) for H=192 and H=336 with stride=96
# This gives fair comparison against unified_v13 which uses stride=96

RESULTS="sweep_results.csv"

# First, remove the old stride=H entries for these horizons
for h in 192 336; do
    sed -i "/,sparsegpt,$h,/d" "$RESULTS"
    sed -i "/,magnitude,$h,/d" "$RESULTS"
done
echo "Deleted old baseline entries for H=192 and H=336"

DATASETS="ETTm1 ETTm2 ETTh1 ETTh2"
HORIZONS="192 336"
CONTEXTS="512 1024 2048"
METHODS="sparsegpt magnitude"

declare -A TRAIN_ENDS
TRAIN_ENDS["ETTm1"]=49152
TRAIN_ENDS["ETTm2"]=49152
TRAIN_ENDS["ETTh1"]=8640
TRAIN_ENDS["ETTh2"]=8640

for ds in $DATASETS; do
    end=${TRAIN_ENDS[$ds]}
    csv="ETDataset/ETT-small/$ds.csv"
    for m in $METHODS; do
        for h in $HORIZONS; do
            for c in $CONTEXTS; do
                if ! grep -q "$ds,$m,$h,$c," "$RESULTS"; then
                    echo "Running: $ds $m H=$h C=$c (stride=96)..."
                    timeout 6000 micromamba run -n timesfm311 python baselines_2of4.py \
                        --csv "$csv" --train_end "$end" --horizon "$h" --context "$c" \
                        --mode "$m" --stride_test 96 > temp_baseline.txt 2>&1

                    # Parse MSE from output
                    mse=$(grep "MSE=" temp_baseline.txt | tail -n 1 | sed 's/.*MSE=//' | awk '{print $1}')

                    if [ -n "$mse" ]; then
                        echo "$ds,$m,$h,$c,$mse" >> "$RESULTS"
                        echo "  -> MSE=$mse"
                    else
                        echo "  -> ERROR: MSE not found"
                        echo "--- FAILED: $ds $m H=$h C=$c ---" >> baseline_fail.log
                        cat temp_baseline.txt >> baseline_fail.log
                    fi
                else
                    echo "Skipping: $ds $m H=$h C=$c (already done)"
                fi
            done
        done
    done
done

echo "Baseline Re-run Complete!"
