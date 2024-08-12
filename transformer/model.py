'''
@autheor: Destin Niyomufasha. Transformer model
July 2024 
'''
import torch 
import torchvision
from torch import nn
import numpy as np
import math


class PositionalEmbddings(nn.Module):p
    '''
    1000 can be changed
    '''
    def __init__(self, sequence_length, dimension):

        '''
        Sequence_length : maximum tokens length
        '''
        super().__init__()
        self.positions = torch.arange(0, sequence_length)
        self.indices = torch.arange(0, dimension, 2) 
        self.position_embeddings = torch.zeros((self.sequence_length, self.dimension))
        self.position_embeddings[:, 0::2] = torch.sin((self.positions) / 1000 ** (self.indices // self.dimension))
        self.position_embeddings[:, 1::2] = torch.cos((self.positions) / 1000 ** (self.indices // self.dimesion))
    def forward(self, x):
        '''
        dimsensions: 
            (batch_size, sequence_length, dimension)
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
    def __init__(self, vocab_size, dimension, max_sequence, dropout):
        self.TokenEmbedding = TokenEmbeddings(vocab_size, dimension)
        self.PositionalEmbedding = PositionalEmbddings(max_sequence, dimension)
        self.dropout_final = nn.Dropout(dropout)
        self.dropout_token = nn.Dropout(dropout)

    def forward(self, x):

        token_embeddings = self.TokenEmbeddings(x)
        position_embeddings = self.dropout_pos(self.PositionalEmbeddings(x))
        
        return self.dropout_final(token_embeddings  + position_embeddings)

class Attention(nn.Module):
    def __init__(self):
        super.__init__()
        '''
        Input dimemsions :
            0 : batch_size
            1 : num_heads
            2 : sequence_length
            3 : token embeddings dimension
        '''

    def forward(self, queries, keys, values, mask = None):
        dimension = values.size(3) 
        transposed_keys = keys.transpose(2, 3) # dimensions [batch_size, num_heads, emb_dim., seq_length]
        attention_score = torch.softmax((queries @ transposed_keys) / math.sqrt(dimension)) # resulting dimention: batch, heads, seq_length, seq_length 
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -100000)

        return torch.matmul(attention_score, values), attention_score



class LayerNorm(nn.Module):
    def __init___(self, dimension, epsilon):
        super.__init__()
        self.gamma = torch.tensor(dimension)
        self.beta = torch.tensor(dimension)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        variance = x.var(-1, keepdim = True)

        returning = ((x - mean) / variance + self.epsilon) * self.gamma + self.beta
    
        return returning
    


class MultiHeadAttention(nn.Module):
    '''
    Input: 
    '''
    def __init__(self, dimension):
        super.__init__()

        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linaer(dimension, dimension)
        self.value_projection = nn.Linear(dimension, dimension)
        self.after_concat_projection = nn.Linear(dimension, dimension)
        self.attention_layer = Attention()

    def partition_inputs(inputs, num_heads):
        '''
        [batch, length, dimension]
        '''
        batch_size, seq_length, dimension = input.size() 
        inputs = inputs.view(batch_size, seq_length, num_heads, dimension / num_heads).transpose(1, 2)
        return inputs
    

    def concanate_ouputs(output):
        '''
        [bactch, heads, length, domension/heads]'''
        batch_size, heads, seq_length, dimension = output.size() 
        model_dimension = dimension/heads
        output_concat = output.transpose(2, 3).view(batch_size, seq_length, model_dimension)
        return output_concat

    def forward(self, queries, keys, values, mask):
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        output, attention_score = self.attention_layer(queries, keys, values, mask)
        
        return self.after_concat_projection(output)
    
class positionwiseFeedforward(nn.Module):
    def __init__(self, dimension, hidden, dropout):
        super.__init__()
        self.Linear1 = nn.Linear(dimension, hidden)
        self.Linear2 = nn.Linear(hidden, dimension)
        self.relu = nn.ReLu()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.Linear2(self.dropout(self.relu(self.Linear1(x))))

        return x
    

class Encoder(nn.Module):
    def __init__(self, max_sequence_length, dimension, vocab_size, hidden, dropout):
        super.__init__()
        self.transformer_embedding = TransformerEmbedding(vocab_size,dimension, max_sequence_length, dropout)
        self.norm1 = LayerNorm(dimension, 1e-12)
        self.norm2 = LayerNorm(dimension, 1e-12)
        self.attention = MultiHeadAttention(dimension)
        self.feedforward = positionwiseFeedforward(dimension, hidden, dropout)

    def forward(self, x):
        embeddings = self.transformer_embedding(x)
        attention = self.attention(embeddings)
        normalized_embeddings = self.norm1(attention) + embeddings   
        output = self.norm2(self.feedforward(normalized_embeddings) )+ normalized_embeddings

        return output
    


class Decoder(nn.Module):
    def __init__(self, max_sequence_length, dimension, vocab_size, hidden, dropout):
        super(Decoder, self).__init__() 
        self.norm1 =  LayerNorm(dimension, 1e-12)
        self.norm2 = LayerNorm(dimension, 1e-12)
        self.norm3 = LayerNorm(dimension, 1e-12)

        self.input_attention = MultiHeadAttention(dimension)
        self.encoder_attention = MultiHeadAttention(dimension)
        self.feedforward = self.feedforward(dimension, hidden,dropout)

        self.transformer_embeddings_decoder = TransformerEmbedding(vocab_size, dimension, max_sequence_length)
        self.projecttion_layer = nn.Linear(dimension, dimension)

    def forward(self, encoder_output, decoder_input):

        x = self.transformer_embeddings_decoder(decoder_input)
        encoder_attention_score = self.encoder_attention(x, x, x)
        x_transformed = self.norm1(x + encoder_attention_score)

        enc_ = encoder_output  + x_transformed

        encoder_transformed = self.encoder_attention(enc_, enc_, enc_)
        x_ = self.norm2(encoder_transformed + x_transformed)

        x_output = self.norm3(self.feedforward(x_) + x_)

        return nn.softmax(self.projecttion_layer(x_output))
    


    class Transformer(nn.Module):
        def __init__(self, max_sequence_length, dimension, vocab_size, hidden, dropout):
            super.__init__()
            self.encoder = Encoder(max_sequence_length, dimension, vocab_size, hidden, dropout)
            self.decoder = Decoder(max_sequence_length, dimension, vocab_size, hidden, dropout)

        def forward()