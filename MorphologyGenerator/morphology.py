import math

def arithmetic_mean(values):
    return sum(values) / len(values)

def geometirc_mean(values):
    return math.prod(values) ** (1/len(values))

def harmonic_mean(values):
    replacement = [1/value for value in values]
    return len(replacement) / sum(replacement)

