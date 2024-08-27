#!/bin/bash

# Load the FST and apply each word
foma -l ../kin-morph-fst/FST/kinyaFST.foma apply up -e 'input.txt' > output.txt
