Low-resource languages benefit from word-level information (Such as morphemes) to enhance information in their embeddings. Often, the morphology generation systems require morphology disambiguation (Determining) which segmentations generated are the correct ones. 

In this code, I take a Neural Network approach to addressing this problem for Kinyarwanda (My native language).

This code is based on the paper: Kinyarwanda Morphology disambiguation (https://arxiv.org/abs/2203.08459)

## Improvements made: 

- Included neighborhood information in feature extraction
- Embeddings were produced using a physics-inspired superposition of the morphemes embeddings (Tensorized embeddings)
