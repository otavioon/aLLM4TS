#!/bin/bash

echo "file,dataset,config,test_accuracy,runs"  # Print CSV header

grep "accuracy:" logs/classification/*_patch-*.log | awk -F '[: ]' '
{
    file=$1;  # Extract full file path
    acc=$NF;  # Extract accuracy value

    # Extract just the filename (removing the directory structure)
    n = split(file, path_parts, "/");
    filename = path_parts[n];

    # Remove the .log extension
    sub(/\.log$/, "", filename);

    # Extract dataset (everything before "_patch")
    split(filename, parts, "_patch");
    dataset = parts[1];

    # Extract config (everything after dataset name)
    config = substr(filename, length(dataset) + 2);  # +2 to remove underscore after dataset

    sum[filename] += acc;
    count[filename] += 1;
    dataset_map[filename] = dataset;
    config_map[filename] = config;
    file_map[filename] = file;
}
END {
    for (filename in sum) {
        printf "%s,%s,%s,%.6f,%d\n", file_map[filename], dataset_map[filename], config_map[filename], sum[filename] / count[filename], count[filename];
    }
}' | sort
