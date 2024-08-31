#!/bin/bash

# Function to apply FST to an input file and output segmentation to an output file
produce_segmentations() {
    local binFile=$1
    local inputFile=$2
    local outputFile=$3
    local command_type=$4

    echo "$command_type"

    # Use the variable command_type to apply the correct command
    foma -e "load $binFile" -e "$command_type < $inputFile" -e "quit" > "$outputFile"
}


produce_segmentations "mudasobwa.bin" "input_.txt" "segmentation.txt" "apply up"
