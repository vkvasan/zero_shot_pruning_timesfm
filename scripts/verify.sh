#!/bin/bash
# verify_restored_v13.sh

echo "=== Verifying Restored v13 Baseline ==="

# 1. ETTm1 H=96 C=1024 (v13 Historical: ~3.03 MSE)
echo "Running ETTm1 H=96 C=1024..."
micromamba run -n timesfm311 python snr_2of4_signal_noise_ratio2_v1.py \
    --csv ETDataset/ETT-small/ETTm1.csv --col OT --train_end 49152 \
    --horizon 96 --context 1024 --score_mode unified --refit 1 --ridge 1e-5 \
    --max_calls_per_layer 64 --calib_select last --nf_hi 0.0 --error_power 0 --stride_test 96 2>&1 | grep -E "snr-2of4|Winner|nsr"

# 2. ETTh2 H=336 C=1024 (v13 Historical Context: Stable Baseline)
echo "Running ETTh2 H=336 C=1024..."
micromamba run -n timesfm311 python snr_2of4_signal_noise_ratio2_v1.py \
    --csv ETDataset/ETT-small/ETTh2.csv --col OT --train_end 8640 \
    --horizon 336 --context 1024 --score_mode unified --refit 1 --ridge 1e-5 \
    --max_calls_per_layer 64 --calib_select last --nf_hi 0.0 --error_power 0 --stride_test 336 2>&1 | grep -E "snr-2of4|Winner|nsr"

# 3. ETTm1 H=192 C=2048 (Verification of Lockdown at 0.05)
echo "Running ETTm1 H=192 C=2048 (Lockdown Test)..."
micromamba run -n timesfm311 python snr_2of4_signal_noise_ratio2_v1.py \
    --csv ETDataset/ETT-small/ETTm1.csv --col OT --train_end 49152 \
    --horizon 192 --context 2048 --score_mode unified --refit 1 --ridge 1e-5 \
    --max_calls_per_layer 64 --calib_select last --nf_hi 0.0 --error_power 0 --stride_test 192 2>&1 | grep -E "snr-2of4|Winner|nsr"
