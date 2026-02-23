#!/bin/bash
RESULTS="sweep_results.csv"

# Ensure header
if [ ! -f "$RESULTS" ]; then
    echo "dataset,method,horizon,context,mse" > "$RESULTS"
fi

DATASETS="ETTm1 ETTm2 ETTh1 ETTh2"
HORIZONS="96 192 336"
CONTEXTS="512 1024 2048"

# Train Ends mapping
declare -A TRAIN_ENDS
TRAIN_ENDS["ETTm1"]=49152
TRAIN_ENDS["ETTm2"]=49152
TRAIN_ENDS["ETTh1"]=8640
TRAIN_ENDS["ETTh2"]=8640

for ds in $DATASETS; do
    end=${TRAIN_ENDS[$ds]}
    csv="ETDataset/ETT-small/$ds.csv"
    for h in $HORIZONS; do
        for c in $CONTEXTS; do
            # Unified v13 ONLY
            if ! grep -q "$ds,unified_v13,$h,$c," "$RESULTS"; then
                echo "Running: $ds unified_v13 H=$h C=$c..."
                timeout 12000 micromamba run -n timesfm311 python snr_2of4_signal_noise_ratio2_v1.py --csv "$csv" --train_end "$end" --horizon "$h" --context "$c" --score_mode unified --refit 1 > temp_v13.txt 2>&1
                
                # CORRECT PARSING: Capture the absolute MSE from the refit line
                mse=$(grep "\[snr-2of4-refit\] MSE=" temp_v13.txt | sed 's/.*MSE=//' | awk '{print $1}')
                
                if [ -z "$mse" ]; then
                   mse=$(grep -a "MSE=" temp_v13.txt | grep -v "delta" | tail -n 1 | sed 's/.*MSE=//' | awk '{print $1}')
                fi

                if [ -n "$mse" ]; then
                    echo "$ds,unified_v13,$h,$c,$mse" >> "$RESULTS"
                    echo "  -> MSE=$mse (v13)"
                else
                    echo "  -> ERROR: MSE not found"
                    echo "--- FAILED: $ds H=$h C=$c ---" >> v13_fail.log
                    cat temp_v13.txt >> v13_fail.log
                fi
            else
                echo "Skipping: $ds unified_v13 H=$h C=$c (done)"
            fi
        done
    done
done

echo "Sweep Complete."
