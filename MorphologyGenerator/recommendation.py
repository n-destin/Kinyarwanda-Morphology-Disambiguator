import math
import numpy as np

# Dummy word embedding lookup function (replace with actual embeddings)
def e(word):
    # Replace with actual word embeddings retrieval
    return np.random.rand(300)  # Assume 300-dimensional embeddings for example

# Normalizing sigmoid function
def sigmoid(z, minf, maxf):
    return 1 / (1 + math.exp(-8 * (z - minf) / (maxf - minf)))

# Function to calculate dot product of two vectors
def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

# Function to calculate the magnitude (norm) of a vector
def magnitude(vec):
    return math.sqrt(dot_product(vec, vec))

# Normalized angular similarity between word embedding vectors
def de(x, yi):
    ex = e(x)
    eyi = e(yi)
    
    dot_prod = dot_product(ex, eyi)
    mag_ex = magnitude(ex)
    mag_eyi = magnitude(eyi)
    
    cosine_similarity = dot_prod / (mag_ex * mag_eyi)
    angular_similarity = math.acos(cosine_similarity) / math.pi
    
    return sigmoid(1 - angular_similarity, minf=0, maxf=1)

# Feature mean functions
def arithmetic_mean(values):
    return sum(values) / len(values)

def geometric_mean(values):
    product = 1
    for value in values:
        product *= value
    return product ** (1 / len(values))

def harmonic_mean(values):
    return len(values) / sum(1.0 / value for value in values)

# Function to compute dt feature
def dt(x, y_list):
    tc_x, td_x = token_doc_frequency(x)
    dt_values = []
    for yi in y_list:
        tc_yi, td_yi = token_doc_frequency(yi)
        dist = math.sqrt((tc_x - tc_yi) ** 2 + (td_x - td_yi) ** 2)
        dt_values.append(dist)
    return sigmoid(np.mean(dt_values), minf=0, maxf=1)

# Function to compute token count and document count normalized by sigmoid
def token_doc_frequency(z):
    # Replace with actual corpus statistics retrieval
    token_count = np.random.randint(1, 1000)  # Dummy value
    doc_count = np.random.randint(1, 100)     # Dummy value
    return sigmoid(token_count, minf=0, maxf=1000), sigmoid(doc_count, minf=0, maxf=100)

# Compute morphological indicator features
def morphological_indicator_features(stem, candidate_features, training_data_features):
    features = []
    for fi in candidate_features:
        chosen = training_data_features.get((fi, stem), {}).get('chosen', 0)
        proposed = training_data_features.get((fi, stem), {}).get('proposed', 1)  # Avoid division by zero
        selection_ratio = chosen / proposed
        features.append(sigmoid(selection_ratio, minf=0, maxf=1))
    return features

# Main function to extract features
def extract_features(x, stem, inflections, common_affixes, training_data_features):
    # Generate possible inflections
    y_candidates = [stem + affix for affix in common_affixes]
    
    # Filter inflections that are in vocabulary
    y_infl = [yi for yi in y_candidates if yi in inflections]
    
    # Calculate similarities and select top K nearest inflections
    similarities = [(yi, de(x, yi)) for yi in y_infl]
    similarities = sorted(similarities, key=lambda item: item[1], reverse=True)
    top_k = similarities[:K]  # Assume K is predefined
    
    # Calculate mean features for the top K
    similarity_scores = [sim for yi, sim in top_k]
    features = {
        'arithmetic_mean_similarity': arithmetic_mean(similarity_scores),
        'geometric_mean_similarity': geometric_mean(similarity_scores),
        'harmonic_mean_similarity': harmonic_mean(similarity_scores)
    }
    
    # Calculate dt feature
    features['dt'] = dt(x, [yi for yi, sim in top_k])
    
    # Calculate token and document frequency-based features
    tc_x, td_x = token_doc_frequency(x)
    tc_td_x = (tc_x + td_x) / 2
    features['tc_td_x'] = tc_td_x
    tc_td_yi = [token_doc_frequency(yi) for yi, sim in top_k]
    features['fm_tc_td_yi'] = arithmetic_mean([np.mean(tc_td) for tc_td in tc_td_yi])
    
    # Morphological indicator features
    candidate_features = ['passivization', 'transitivity']  # Example morphological features
    indicator_features = morphological_indicator_features(stem, candidate_features, training_data_features)
    
    return features, indicator_features

# Example usage
stem = "run"
x = "running"
inflections = {"runs", "ran", "running", "runner"}  # Vocabulary set
common_affixes = {"s", "ing", "er"}  # Common inflections/affixes
training_data_features = {
    ('passivization', 'run'): {'chosen': 10, 'proposed': 15},
    ('transitivity', 'run'): {'chosen': 5, 'proposed': 10}
}

features, indicator_features = extract_features(x, stem, inflections, common_affixes, training_data_features)

print("Extracted Features:", features)
print("Morphological Indicator Features:", indicator_features)
