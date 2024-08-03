'''
@autheor: Destin Niyomufasha. Transformer model
July 2024 
'''
import torch 
import torchvision
from torch import nn
import numpy as np
import math


class PositionalEmbddings(nn.Module):
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
                
class TokenEmbeddings(nn.Module):
    '''
    Returns an embedding matrix
    '''
    def __init__(self, vocab_size, dimension):
        super(TokenEmbeddings, self).__init__(vocab_size, dimension)



class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, dimension, x, max_sequence, dropout):
        self.TokenEmbedding = TokenEmbeddings(vocab_size, dimension)
        self.PositionalEmbedding = PositionalEmbddings(max_sequence, dimension)
        self.dropout_final = nn.Dropout(dropout)
        self.dropout_token = nn.Dropout(dropout)

        def forward(self, x):

            token_embeddings = self.TokenEmbeddings(x)
            position_embeddings = self.dropout_pos(self.PositionalEmbeddings(x))
            
            return self.dropout_final(token_embeddings  + position_embeddings)

class Attention(nn.Module):
    def __init__(self, query, key, value, dimension):
        super.__init__()
        '''
        Input dimemsions :
            0 : batch_size
            1 : num_heads
            2 : sequence_length
            3 : token embeddings dimension
        '''
        transposed_keys = key.transpose(2, 3)

        def forward(self):
            return torch.sigmoid(torch.matmul(self.quey, self.key / math.sqrt(self.dimesion))) * self.value
        


class MultiHeadAttention(nn.Module):
    '''
    Input: 
    '''
    def __init__(self, num_heads, dimension, query, key, value):
        super.__init__()

        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linaer(dimension, dimension)
        self.value_projection = nn.Linear(dimension, dimension)
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