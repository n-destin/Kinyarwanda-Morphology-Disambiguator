'''
@Author: Destin Niyomufasha
Intelligence and Signal Processing Lab
'''

import subprocess
from transition import TransitionGraph

graph = TransitionGraph("file here")

def generate_segmentations(bash_function, bin_file, inputFile, outputFile, input_command):
    command = f'source applyFST.sh && {bash_function} {bin_file} {inputFile} {outputFile} "{input_command}"'
    subprocess.run(['bash', '-c', command], capture_output=True, text=True)

def generate_inflections()

generate_segmentations("produce_segmentations", "mudasobwa.bin", "input_.txt", "segmentation.txt", "apply up")