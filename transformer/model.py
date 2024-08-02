'''
@autheor: Destin Niyomufasha. Transformer model
July 2024 
'''


import torch 
import torchvision
from torch import nn
import numpy as np
import math


class Positional_encoder(nn.Module):
    '''
    1000 can be changed
    '''
    def __init__(self, sequence_length, dimension):
        super().__init__()
        self.positions = torch.arange(0, sequence_length)
        self.indices = torch.arange(0, dimension, 2) 
        self.position_embeddings = torch.zeros((self.sequence_length, self.dimension))
        self.position_embeddings[:, 0::2] = torch.sin((self.positions) / 1000 ** (self.indices // self.dimension))
        self.position_embeddings[:, 1::2] = torch.cos((self.positions) / 1000 ** (self.indices // self.dimesion))
    def forward(self, x):
        '''
        
        '''
        batch_size, seq_length = x.size()

        return self.position_embeddings[:seq_length, :]
                
class Embedding(nn.Module):
    '''
    Returns an embedding matrix
    '''
    def __init__(self, vocab_size, dimension, dropout):
        super.__init__()
        self.vocab_size = vocab_size
        self.dimension = dimension
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embeddings(self.vocab_size, self.dimension)

        def forward(index):
            return self.dropout(self.embeddings[index])

class Attention(nn.Module):
    def __init__(self, query, key, value, dimension):
        super.__init__()
        self.query =  query
        self.key = key
        self.value = value
        self.dimension = dimension

        def forward(self):
            return torch.sigmoid(torch.matmul(self.quey, self.key / math.sqrt(self.dimesion))) * self.value
        


class MultiHeadAttention(nn.Module):
    '''
    Input: 
    '''
    def __init__(self, num_heads, dimension, query, key, value):
        super.__init__()
        self.num_heads = num_heads
        self.dimension = dimension
        self.query = query
        self.key = key
        self.value = value
        self.query_projection = nn.Linear(self.dimension, self.dimension)
        self.key_projection = nn.Linear(self.dimension, self.dimension)
        self.value_projection = nn.Linear(self.dimension, self.dimension)
        self.attention_layer = Attention()

    def forward(self):
        final_attention = torch.zeros(self.dimension)
        for head in self.num_heads:
            query = self.query_projection(self.query)
            key = self.key_projection(self.key)
            value = self.value_projection(self.value)
            attention_value = self.attention_layer(query, key, value)
            final_attention.apend(attention_value)

        
        return final_attention
    



class Encoder(nn.Module):
    def __init__(self, sequence, dimension, vocab_size):
        super.__init__()
        self.dimension = dimension
        self.vocal_size = vocab_size
        self.sequence = sequence
        self.embedding = Embedding(self.vocab_size, self.dimension)

