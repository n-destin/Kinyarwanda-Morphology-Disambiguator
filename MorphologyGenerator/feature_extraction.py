'''
@author: Destin Niyomufahsa
@Morphologty analyzer
Get the embeddings from KinNews and KirNews
'''


from generate import generate_inflections
from feature_lib import arithmetic_mean, geometirc_mean, harmonic_mean, normalized_angular_similarity, sigmoid, embeddings
from train_transition_graph import generate_segmentations, graph

def words_roots_processing(input_file, wordsFile):
    mapping = {}
    with open(input_file, "r") as file:
        with open(wordsFile, "w") as segmentationFile:
            lines = file.readlines()
            for line in lines:
                word, root = line.split(",")
                segmentationFile.write(word + "\n")
                if word[-1] == "\n":
                    word = word[0:len(word) - 1]
                mapping[word] = [root[0:len(root) - 1]]
    return mapping

roots_reference = words_roots_processing("word_to_root.txt", "wordsFile.txt")
# print(roots_reference)

def moprh_dataset(words_file, segmentationFile):
    '''
    This function This constructs a database of the times a root was proposed, and the time it was selected, and the times it was selected with certain morpohologies
    '''
    generate_segmentations("produce_segementations", "compiled.bin", words_file, segmentationFile, "apply up")
    proposed = {}
    new_word = True
    chosen = {}
    current = ""
    with open(segmentationFile, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            root = None
            if new_word:
                current  = line[0:len(line) - 1]
                new_word = False
            if line == "\n":
                new_word = True
            elif len(line) > 1 and line[0] == "+":
                morphemes = line.split("+")
                for morpheme in morphemes:
                    if morpheme.islower():
                        root = morpheme
                for morpheme in morphemes:
                    # root -> morphemes ->number of times together
                    if root in proposed.keys():
                        if morpheme in proposed[root].keys():
                            proposed[root][morpheme] += 1
                        else:
                            proposed[root][morpheme] = 1
                    else:
                        proposed[root] = {morpheme : 1}
                        # root ->morpheme -> number of times chosen together
                    if root == roots_reference[current]:
                        if root in chosen.keys():
                            if morpheme in chosen[root].keys():
                                chosen[root][morpheme] += 1
                            else:
                                chosen[root][morpheme] = 1
                        else:
                            chosen[root] = {morpheme : 1}
                if current != "":
                    print(root, roots_reference[current], current)

    return proposed, chosen


def generate_mean(measure):
    return [arithmetic_mean(measure), harmonic_mean(measure), geometirc_mean(measure)]

# we also need to know the morphology segmentation of the words
def feature_extraction(words):
    similarity_features = {} # word - > similarity
    to_return = {}
    inflections, words_segmentations = generate_inflections(words, "transition_input.txt", "transition_output.txt", "final_output_txt")
    #proposed and chosen: word -> word - > root -> count
    proposed, chosen = moprh_dataset("wordsFile.txt", "segmentations.txt")
    
    for word in inflections.keys():
        real_segmentation = None
        to_return[word] = {}
        for root in inflections[word]:
            measure = []
            for inflection in inflections[word][root]:
                inflection_embedding = embeddings[inflection] if inflection in embeddings.keys() else [0] * 100
                word_embedding = embeddings[word] if word in embeddings.keys() else [0] * 100
                measure.append(sigmoid(normalized_angular_similarity(inflection_embedding, word_embedding)))
            # print(measure)
            similarity_features[root] = generate_mean(measure)
        for segmentation in words_segmentations[word]:
            root = ""
            for morpheme in segmentation.split("+"):
                if morpheme.islower():
                    root = morpheme
                    break
            features = [0] * (graph.morpheme_count + 1)
            for morpheme in segmentation.split("+"):
                if morpheme.isupper():
                    print(morpheme, morpheme.isupper(), similarity_features)
                    features[graph.morpheme_mapping[morpheme]] = sigmoid(chosen[root][morpheme])
                    features += similarity_features[root]
            features = features + generate_mean([proposed[root][morpheme] for morpheme in segmentation.split("+")])
            to_return[word][segmentation] = features
    
    return to_return


print(feature_extraction(["ukabize", "rwitangiraga", "rutanguye", "rizahuriza", "ntirubahiriza", "ntimurakababarirwa", "kwariciye", "kubiteza","batarariraga"]))