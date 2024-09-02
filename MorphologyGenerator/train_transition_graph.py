'''
@Author: Destin Niyomufasha
Intelligence and Signal Processing Lab
'''

import subprocess
from transition import TransitionGraph


def generate_segmentations(bash_function, bin_file, inputFile, outputFile, input_command):
    command = f'source applyFST.sh && {bash_function} {bin_file} {inputFile} {outputFile} "{input_command}"'
    subprocess.run(['bash', '-c', command], capture_output=True, text=True)

generate_segmentations("produce_segmentations", "compiled.bin", "corpus.txt", "output_transition.txt", "apply up")
graph = TransitionGraph("output_transition.txt")
# graph.print_transition_graph()
# print(graph.get_inflections("tungur"))