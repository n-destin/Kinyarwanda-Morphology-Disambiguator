import torch 
from torch import nn
from transformer.transformer import MultiHeadAttention
from transformer.transformer import TokenEmbeddings


class SegmentEmbedding(nn.Module):
    def __init__(self, embedding_dimension):
        super.__init__()
        self.embedding_dimension = embedding_dimension
        self.embeddings = nn.Embedding(2, embedding_dimension)

    def forward(self, x):
        input_length = len(x)
        for index in range(len(x)):
            if x[index] == 'SEP':
                input_length = index
        to_return = torch.tensor(len(x), self.embedding_dimension)
        to_return[:input_length, :] = self.embeddings[0]
        to_return[input_length : , :] = self.embeddings[1]

        return to_return 

class BERTPLain(nn.Module):
    def __init__(self,vocab_size, embedding_dimension, max_length):
        super.__init__()
        self.TokenEmbeddings = TokenEmbeddings(vocab_size, embedding_dimension)
        self.positionEmbeddings = nn.Embedding(max_length, embedding_dimension)
        self.segmentEmbeddings = SegmentEmbedding(embedding_dimension)


