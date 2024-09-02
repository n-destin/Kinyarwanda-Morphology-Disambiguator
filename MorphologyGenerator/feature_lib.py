import math
import numpy as np

def arithmetic_mean(values):
    if len(values) == 0:
        return 0
    return sum(values) / len(values)

def geometirc_mean(values):
    if len(values) == 0:
        return 0
    return math.prod(values) ** (1/len(values))

def harmonic_mean(values):
    if len(values) == 0:
        return  0
    replacement = [1/value for value in values]
    return len(replacement) / sum(replacement)

def sigmoid(value):
    return 1 / (1 + math.exp(value))

def load_embeddings(embeddings_file):
    to_return = {}
    with open(embeddings_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            word_embedding = line.split(' ')
            to_return[word_embedding[0]] = word_embedding[1:len(word_embedding)]
    
    return to_return


def normalized_angular_similarity(embedding_one, embedding_two):
    value = (np.array(embedding_one) * np.array(embedding_two))[0] /  (len(embedding_one) + len(embedding_two))
    return value

embeddings = load_embeddings("./embeddings/embeddings.txt")