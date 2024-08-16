import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # d_in: input dimension
        # d_out: output dimension (also used as the attention dimension)
        self.d_in = d_in
        self.d_out = d_out
        
        # Linear transformations for Query, Key, and Value
        self.Q = nn.Linear(d_in, d_out)
        self.K = nn.Linear(d_in, d_out)
        self.V = nn.Linear(d_in, d_out)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_in)
        
        # Transform input to Query, Key, and Value
        queries = self.Q(x)  # shape: (batch_size, seq_len, d_out)
        keys = self.K(x)     # shape: (batch_size, seq_len, d_out)
        values = self.V(x)   # shape: (batch_size, seq_len, d_out)
        
        # Compute attention scores
        # torch.bmm performs batch matrix multiplication
        scores = torch.bmm(queries, keys.transpose(1, 2))
        # shape: (batch_size, seq_len, seq_len)
        
        # Scale the scores
        scores = scores / (self.d_out ** 0.5)  # Apply scaling factor
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=2)
        # shape: (batch_size, seq_len, seq_len)
        
        # Compute the weighted sum of values
        hidden_states = torch.bmm(attention, values)
        # shape: (batch_size, seq_len, d_out)
        
        return hidden_states
    

class BasicMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size  # Total hidden size
        self.num_heads = num_heads      # Number of attention heads
        
        # Linear layer to combine outputs from all heads
        self.out = nn.Linear(hidden_size, hidden_size)
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            Attention(hidden_size, hidden_size // num_heads) 
            for _ in range(num_heads)
        ])
        # Each head operates on a slice of the hidden state
        # hidden_size // num_heads is the size of each head's output
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        
        # Apply each attention head to the input
        outputs = [head(x) for head in self.heads]
        # Each output shape: (batch_size, seq_len, hidden_size // num_heads)
        
        # Concatenate the outputs from all heads
        outputs = torch.cat(outputs, dim=2)
        # shape after concatenation: (batch_size, seq_len, hidden_size)
        
        # Apply the output linear transformation
        hidden_states = self.out(outputs)
        # shape: (batch_size, seq_len, hidden_size)
        
        return hidden_states


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Single linear layer for Q, K, V projections
        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        # Output projection
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        
        # Project input to Q, K, V
        # (batch_size, seq_length, hidden_size * 3)
        qkv = self.qkv_linear(x)
        
        # Reshape and transpose for multi-head attention
        # (batch_size, seq_length, num_heads, head_dim * 3)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        # (batch_size, num_heads, seq_length, head_dim * 3)
        qkv = qkv.transpose(1, 2)
        
        # Split into Q, K, V
        # Each of shape (batch_size, num_heads, seq_length, head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_length, seq_length)
        scores = torch.matmul(queries, keys.transpose(2, 3))
        
        # Scale scores
        # (batch_size, num_heads, seq_length, seq_length)
        scores = scores / (self.head_dim ** 0.5)
        
        # Apply softmax to get attention weights
        # (batch_size, num_heads, seq_length, seq_length)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_length, head_dim)
        context = torch.matmul(attention, values)
        
        # Transpose and reshape to combine heads
        # (batch_size, seq_length, num_heads, head_dim)
        context = context.transpose(1, 2)
        # (batch_size, seq_length, hidden_size)
        context = context.reshape(batch_size, seq_length, hidden_size)
        
        # Apply output projection
        # (batch_size, seq_length, hidden_size)
        output = self.out(context)
        
        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model):
        super().__init__()
        
        # Create a matrix of shape (context_size, d_model) to store the positional encodings
        self.encoding = torch.zeros(context_size, d_model)
        
        # Create a tensor of positions from 0 to context_size - 1
        # Shape: (context_size, 1)
        pos = torch.arange(0, context_size).unsqueeze(dim=1)
        
        # Create a tensor of even indices from 0 to d_model - 2
        # Shape: (d_model / 2)
        dim = torch.arange(0, d_model, 2)
        
        # Compute the arguments for the sine and cosine functions
        # Shape: (context_size, d_model / 2)
        arg = pos / (10000 ** (dim / d_model))
        
        # Compute sine values for even indices
        self.encoding[:, 0::2] = torch.sin(arg)
        
        # Compute cosine values for odd indices
        self.encoding[:, 1::2] = torch.cos(arg)

        self.register_buffer('encoding', self.encoding)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Return the positional encoding for the given sequence length
        return self.encoding[:, :seq_len, :]
    

class EfficientPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a tensor of positions from 0 to max_len - 1
        # Shape: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        
        # Compute the division terms for the encoding
        # This is an optimization of the original formula
        # Shape: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize the positional encoding tensor
        # Shape: (max_len, 1, d_model)
        pe = torch.zeros(max_len, 1, d_model)
        
        # Compute sine values for even indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        # Compute cosine values for odd indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding as a buffer
        # This means it won't be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Return the positional encoding for the given sequence length
        return self.encoding[:, :seq_len, :]
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # First linear transformation
        # Increases dimensionality from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear transformation
        # Decreases dimensionality back from d_ff to d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        
        # Apply first linear transformation
        x = self.linear1(x)
        
        # Apply ReLU activation
        x = self.relu(x)
        
        # Apply second linear transformation
        x = self.linear2(x)
        
        # Output shape: (batch_size, seq_len, d_model)
        return x