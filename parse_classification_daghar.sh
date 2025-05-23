#!/bin/bash

# Get arguments
log_dir="${1:-logs/classification}"
output_csv="${2:-allm4ts_results_daghar.csv}"
tmp_dir=$(mktemp -d)
tmp_file="$tmp_dir/$output_csv"

# Functions
remove_temp_dir() {
    if [[ -d "$tmp_dir" ]]; then
        rm -rf "$tmp_dir"
        echo "Temporary directory removed: $tmp_dir"
    fi
}

die() {
    echo "Error: $1"
    remove_temp_dir
    exit 1
}

echo "--------------------------------"
echo "Running all scripts to parse logs and convert to CSV..."
echo "LOG_DIR: $log_dir"
echo "OUTPUT_CSV: $output_csv"
echo "TEMP FILE: $tmp_file"
echo "--------------------------------"


echo "Parsing classification logs..."
python parse_logs.py --log_dir ${log_dir} --agg --output_file ${tmp_file}  || die "Failed to parse classification logs"

echo "Converting CSV to analysis format..."
python convert_csv_to_analysis.py --input_csv ${tmp_file} --output_csv ${output_csv} || die "Failed to convert CSV to analysis format"

echo "Removing temporary directory..."
remove_temp_dir

echo ""
echo "Parsed results saved to $output_csv"

