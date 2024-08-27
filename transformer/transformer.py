'''
@Author: Destin Niyomufasha. Intelligence and Signal Processing Lab. 
@KinyaBERT
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
        dimsensions: (batch_size, sequence_length, dimension)
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
    def __init__(self, dimension, num_heads):
        super.__init__()


        self.num_heads = num_heads
        self.dimension = dimension

        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linaer(dimension, dimension)
        self.value_projection = nn.Linear(dimension, dimension)

        self.after_concat_projection = nn.Linear(dimension, dimension)
        self.attention_layer = Attention()

    def partition_inputs(self, inputs):
        '''
        [batch, length, dimension]
        '''
        batch_size, seq_length, dimension = input.size() 
        inputs = inputs.view(batch_size, seq_length, self.num_heads, dimension / self.num_heads).transpose(1, 2)
        return inputs
    

    def concanate_ouputs(self, output):
        '''
        [bactch, heads, length, domension/heads]'''
        batch_size, heads, seq_length, dimension = output.size() 
        model_dimension = dimension * heads
        output_concat = output.transpose(2, 3).view(batch_size, seq_length, model_dimension)
        return output_concat

    def forward(self, queries, keys, values, mask):
        queries = self.partition_inputs(self.query_projection())
        keys = self.partition_inputs(self.key_projection(keys))
        values = self.partition_inputs(self.value_projection(values))

        output, attention_score = self.attention_layer(queries, keys, values, mask)
        
        output_ = self.concanate_ouputs(output)
        return self.after_concat_projection(output_)
    
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
    

class EncoderLayer(nn.Module):
    def __init__(self, dimension, hidden, num_heads, dropout):
        super.__init__()
        self.norm1 = LayerNorm(dimension, 1e-12)
        self.norm2 = LayerNorm(dimension, 1e-12)
        self.attention = MultiHeadAttention(dimension, num_heads)
        self.feedforward = positionwiseFeedforward(dimension, hidden, dropout)

    def forward(self, x, mask):
        attention = self.attention(x, mask)
        normalized_embeddings = self.norm1(attention + x)   
        output = self.norm2(self.feedforward(normalized_embeddings) + normalized_embeddings)

        return output
    


class DecoderLayer(nn.Module):
    def __init__(self, dimension, hidden, num_heads, dropout, mask):
        super(DecoderLayer, self).__init__() 
        self.norm1 =  LayerNorm(dimension, 1e-12)
        self.norm2 = LayerNorm(dimension, 1e-12)
        self.norm3 = LayerNorm(dimension, 1e-12)

        self.encoder_attention = MultiHeadAttention(dimension, num_heads)
        self.decoder_attention = MultiHeadAttention(dimension, num_heads,)
        self.feedforward = self.feedforward(dimension, hidden, dropout)

        self.projecttion_layer = nn.Linear(dimension, dimension)

    def forward(self, encoder_output, decoder_input):
        
        decoder_attention_score = self.decoder_attention(decoder_input, decoder_input, decoder_input)
        x_transformed = self.norm1(decoder_input + decoder_attention_score)

        encoder_transformed = self.encoder_attention(x_transformed, encoder_output, encoder_output)
        x_ = self.norm2(encoder_transformed + x_transformed)

        x_output = self.norm3(self.feedforward(x_) + x_)

        return x_output

class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, vocab_size, max_sequence, d_model, hidden_dim, num_layers, dropout_prob):
        super.__init__()
        self.embeddings = TransformerEmbedding(vocab_size, 
                                                d_model,
                                                max_sequence, 
                                                dropout_prob)
        self.Encoder = nn.ModuleList(EncoderLayer(d_model, hidden_dim, num_heads, dropout_prob) for _ in range(num_layers))

    def forward(self, x):
        x_ = self.embeddings(x)
        x = x_
        for encoder in self.Encoder:
            x_ = encoder(x)
            x = x_

        return x
        
class TransfromerDecoder(nn.Module):
    def __init__(self, dec_vocab_size, dec_max_length, hidden_dim, num_heads, dimension, num_layers, decoder_mask, dropout_prob):
        super.__init__()
        self.decoder_embeddings = TransformerEmbedding(dec_vocab_size, dimension, dec_max_length, dropout_prob)
        self.DecoderLayers = nn.ModuleList([DecoderLayer(dimension, hidden_dim, num_heads, dropout_prob, decoder_mask)] for _ in range(num_layers))

    def forward(self, encoder_ouput, decoder_input, target_mask):
        x = decoder_input
        for layer in self.DecoderLayers:
            x_  = layer(encoder_ouput, x)
            x = x_
        
        return x


class Transformer(nn.Module):
    def __init__(self, enc_max_sequence_length, dimension, dec_max_length, dec_vocab_size, enc_vocab_size, hidden, num_heads, num_encoder_layers, num_decoder_layers, x_padding_value, y_padding_value, dropout):
        super.__init__()
        self.x_padding = x_padding_value
        self.y_padding = y_padding_value

        self.encoder = TransformerEncoder(num_heads, 
                                            enc_vocab_size, 
                                            enc_max_sequence_length, 
                                            dimension, 
                                            hidden, 
                                            num_encoder_layers, 
                                            dropout)
        
        self.decoder = TransfromerDecoder(dec_vocab_size, 
                                            dec_max_length, 
                                            hidden, num_heads, 
                                            dimension, 
                                            num_decoder_layers,
                                            dropout)
        
        self.projection_layer = nn.Linear(dimension, dec_vocab_size)

    def forward(self, input_sequence, output_sequence):
        input_mask = self.input_mask(input_sequence)
        target_mask = self.traget_mask(output_sequence)
        encoder_output = self.encoder(input_sequence, input_mask)
        output = self.decoder(encoder_output, output_sequence, target_mask)
        return nn.Softmax(self.projection_layer(output))
    

    def input_mask(self, input):
        mask = torch.tensor(input != self.x_padding).unsqueeze(1).unsqueeze(2)
        return mask

    def traget_mask(self, target, device):
        seq_length = target.shape[1]
        padding_mask = torch.tensor(target != self.y_padding).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones(seq_length, seq_length)).to(device)

        target_mask = padding_mask + subsequent_mask

        return target_mask