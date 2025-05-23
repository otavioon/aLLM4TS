#!/bin/bash

# Get arguments
log_dir="${1:-logs/classification_ts}"
output_csv="${2:-allm4ts_results_ts.csv}"

die() {
    echo "Error: $1"
    remove_temp_dir
    exit 1
}

echo "--------------------------------"
echo "Running all scripts to parse logs and convert to CSV..."
echo "LOG_DIR: $log_dir"
echo "OUTPUT_CSV: $output_csv"
echo "--------------------------------"


echo "Parsing classification logs..."
python parse_logs.py --log_dir ${log_dir} --agg --output_file ${output_csv}  || die "Failed to parse classification logs"

