'''
@author: Destin Niyomufahsa
@Morphologty analyzer

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



def words_to_roots(input_file):
    mapping = {}
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split("::")
            word = process_segementation(parts[0])[0]
            for morpheme in process_segementation(parts[2]):
                for candidate in process_segementation(parts[1]):
                    # print('reached here', morpheme, candidate)
                    if len(morpheme) + 1 == len(candidate) and morpheme == candidate[0 : len(candidate) - 1]:
                        if word in mapping.keys():
                            mapping[word].append(morpheme)
                        else:
                            mapping[word] = [morpheme]
    
    return mapping


def moprh_dataset(file):
    proposed = {}
    new_word = True
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
                    if root in proposed.keys():
                        if morpheme in proposed[root].keys():
                            proposed[root][morpheme] += 1
                        else:
                            proposed[root][morpheme] = 1
                    else:
                        proposed[root] = {morpheme : 1}
    return proposed


returned = words_to_roots("testing.txt")
print(returned)