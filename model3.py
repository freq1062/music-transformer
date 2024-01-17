#Decoder only transformer 2024/01/14
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import get_config
import numpy as np
c = get_config()

class InputEmbeddings(nn.Module):
    #Create an embedding of size d_model for each word in the input sequence
    def __init__(self, d_model: int, vocab_size: int) -> None:
        #Creates embeddings for each word in the vocabulary
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class FeedForwardBlock(nn.Module):
    #Essentially a fully connected layer 
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=256, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len] #Auto mask
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
    
class DecoderBlock(nn.Module):
    #Basically arranges all of the defined layers, and returns the transformed embedding sequences
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = RelativeGlobalAttention(d_model, num_heads, c["seq_len"], dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        attn = self.self_attn(x)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        ff = self.feed_forward(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)

        return x

class MusicTransformer(nn.Module):
    #Arrange all decoder blocks and obtain probability distributions for output
    def __init__(self, vocab_size, d_model:int=512, num_heads:int=8, depth:int=6, d_ff:int=2048, dropout:float=0.1):
        super().__init__()
        self.vocab_size = vocab_size #Number of embeddings to create
        self.d_model = d_model
        self.embed = InputEmbeddings(d_model, vocab_size)
        self.to_scores = nn.Linear(d_model, vocab_size) #For final probabilities
        # Above is the projection layer i just realized
        
        decoder_blocks = []
        for _ in range(depth):
            decoder_blocks.append(DecoderBlock(d_model, num_heads, d_ff, dropout))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        #x: (batch_size, seq_len)
        x = self.embed(x)
        #x: (batch_size, seq_len, d_model)
        for layer in self.decoder_blocks:
            x = layer(x)
        z = self.norm(x)
        return self.to_scores(z)
