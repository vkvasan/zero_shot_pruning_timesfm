#!/bin/bash

# =================================================================
# MASTER BENCHMARK SCRIPT (UPDATED)
# Comparies: MAG vs WANDA vs SPARSEGPT vs UNIVERSAL
# =================================================================

DEVICE="cuda"
LOG_FILE="benchmark_summary_v2.txt"
PYTHON_SCRIPT="chronos_universal_v2.py" # Make sure this matches the new filename

echo "=== CHRONOS UNIVERSAL BENCHMARK (4-WAY) ===" > $LOG_FILE
echo "Date: $(date)" >> $LOG_FILE
echo "-------------------------------------------" >> $LOG_FILE

# Datasets
declare -a datasets=(
    "ETDataset/ETT-small/ETTm1.csv"
    "ETDataset/ETT-small/ETTm2.csv"
    "ETDataset/ETT-small/ETTh1.csv"
    "ETDataset/ETT-small/ETTh2.csv"
    "electricity/electricity.csv"
    "weather/weather.csv"
    "traffic/traffic.csv"
)

for csv_path in "${datasets[@]}"; do
    echo ""
    echo "Running on: $csv_path"
    
    if [ -f "$csv_path" ]; then
        python $PYTHON_SCRIPT --csv_path "$csv_path" --device $DEVICE | tee temp_run.log
        
        # Grab scores
        MAG=$(grep "MAG MSE:" temp_run.log)
        WANDA=$(grep "WANDA MSE:" temp_run.log)
        SGPT=$(grep "SPARSEGPT MSE:" temp_run.log)
        UNIV=$(grep "UNIVERSAL MSE:" temp_run.log)
        CONF=$(grep "Data Periodicity Confidence:" temp_run.log)
        
        echo "" >> $LOG_FILE
        echo "DATASET: $csv_path" >> $LOG_FILE
        echo "$CONF" >> $LOG_FILE
        echo "$MAG" >> $LOG_FILE
        echo "$WANDA" >> $LOG_FILE
        echo "$SGPT" >> $LOG_FILE
        echo "$UNIV" >> $LOG_FILE
        echo "-------------------------------------------" >> $LOG_FILE
    else
        echo "File not found: $csv_path"
    fi
done

echo ""
echo "DONE. Check $LOG_FILE"
rm temp_run.log