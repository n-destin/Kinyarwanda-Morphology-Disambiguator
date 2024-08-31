'''
@author: Destin Niyomufahsa
@Morphologty analyzer
Get the embeddings from KinNews and KirNews
'''


def process_segementation(segementation):
    to_return = []
    current = ""
    for char in segementation:
        if not char.isalpha():
            if len(current) != 0:
                to_return.append(current)
            current = ""
        else:
            current += char
    return to_return


def words_roots_processing(input_file, mapping):
    with open(input_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            segment = line.split(",")
            mapping[segment[0]] = [segment[1][0:len(segment[1]) - 1]]
    return mapping

    
def words_to_roots(input_file, mapping):
    '''
    This function processes a pair of words and its segementation
    '''
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split("::")
            word = process_segementation(parts[0])[0]
            for morpheme in process_segementation(parts[2]):
                for candidate in process_segementation(parts[1]):
                    if len(morpheme) + 1 == len(candidate) and morpheme == candidate[0 : len(candidate) - 1]:
                        if word in mapping.keys():
                            mapping[word].append(morpheme)
                        else:
                            mapping[word] = [morpheme]


def moprh_dataset(file):
    '''
    This function This constructs a database of the times a root was proposed, and the time it was selected, and the times it was selected with certain morpohologies
    '''
    proposed = {}
    new_word = True
    chosen = {}
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            root = None
            if line == "\n":
                new_word = True
            else:
                if new_word:
                    correct_root  = ""
                morphemes = line.split("+")
                for morpheme in morphemes:
                    if morpheme.islower():
                        root = morpheme
                for morpheme in morphemes:
                    # builds the map for the number of times a certain 
                    if root in proposed.keys():
                        if morpheme in proposed[root].keys():
                            proposed[root][morpheme] += 1
                        else:
                            proposed[root][morpheme] = 1
                    else:
                        proposed[root] = {morpheme : 1}
                        #builds the dataset for the main
                    if root in chosen.keys():
                        if morpheme in chosen[root].keys():
                            chosen[root][morpheme] += 1
                        else:
                            chosen[root][morpheme] = 1
                    else:
                        chosen[root] = {morpheme : 1}
    return proposed


worts_to_roots_ = {}
words_roots_processing("word_to_root.txt", worts_to_roots_)
words_to_roots("words1.txt", worts_to_roots_)
print(len(worts_to_roots_))