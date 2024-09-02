from train_transition_graph import graph, generate_segmentations

def write_words(words, writing_file):
    with open(writing_file, "w") as file:
        for word in words:
            file.write(word + "\n")

def get_inflections(segmentation_file):
    to_return = {}
    words_segmentations = {}
    current = ""
    with open(segmentation_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            if len(line) > 0:
                if line[0] == "+":
                    words_segmentations[current].append(line[0:len(line) - 1])
                    for morph in set(line.split("+")):
                        if morph.islower():
                            if morph[-1] == "\n":
                                morph = morph[0:len(morph) - 1]
                            to_return[current][morph] = graph.get_inflections(morph)
                else:
                    current = line[0:len(line) - 1]
                    words_segmentations[current] = []
                    to_return[current] = {}

    return to_return, words_segmentations

def read_into_array(file_name):
    to_return = set()
    with open(file_name, "r") as file:
        for line in file.readlines():
            to_return.add(line[0:len(line) - 1])
    return to_return

def segmentations_to_words(segmentation_file):
    to_return = set()
    with open(segmentation_file, "r") as file:
        for line in file.readlines()[1:]:
            if line[0] != "+" and len(line) > 1 and line[0:len(line) - 1] != "???":
                to_return.add(line[0:len(line) - 1])
    return to_return



def generate_inflections(words, transition_input, transition_output, final_output):
    write_words(words, transition_input)
    generate_segmentations("produce_segmentations", "compiled.bin", transition_input, transition_output, "apply up") # this produces the morphology generations of the words
    words_inflections, words_segmentations = get_inflections(transition_output) # word -> roots
    to_return = {}
    for word in list(words_inflections.keys())[2:]:
        to_return[word] = {}
        for root in words_inflections[word].keys():
            write_words(words_inflections[word][root], transition_input)
            generate_segmentations("produce_segmentations", "compiled.bin", transition_input, final_output, "apply down")
            to_return[word][root] = set(segmentations_to_words(final_output))   
    return to_return, words_segmentations

# returned, morpheme_segmentations = generate_inflections(["yabatunguye", "abaha", "zanditseho", "boroherezwa", "ntibakomeze", "kwimurwa"], "transition_input.txt", "transition_output.txt", "final_output_txt")
# print(returned)