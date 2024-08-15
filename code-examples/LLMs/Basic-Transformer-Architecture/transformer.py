import torch
import torch.nn as nn
import torch.nn.functional as F

from components import (
    PositionalEncoding,
    PositionwiseFeedForward
)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Multi-head Self-Attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # Layer Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        
        # Self-Attention
        # Note: PyTorch's MultiheadAttention expects input in shape (seq_len, batch_size, d_model)
        hidden_states, _ = self.self_attn(query=x, key=x, value=x)
        
        # Add & Norm (Residual connection and Layer Normalization)
        x = self.norm1(x + hidden_states)
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        
        # Add & Norm (Residual connection and Layer Normalization)
        x = self.norm2(x + ff_output)
        
        # Output shape: (seq_len, batch_size, d_model)
        return x
    

class Encoder(nn.Module):
    def __init__(self, input_size, context_size, 
                 d_model, d_ff, num_heads, n_blocks):
        super().__init__()
        
        # Embedding layer to convert input tokens to vectors
        self.embedding = nn.Embedding(input_size, d_model)
        
        # Positional encoding to add position information
        self.pos_embedding = PositionalEncoding(context_size, d_model)

        # Stack of Encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Apply embedding and add positional encoding
        x = self.embedding(x) + self.pos_embedding(x)
        # x shape after embedding: (batch_size, seq_len, d_model)
        
        # Pass through each encoder block
        for block in self.blocks:
            x = block(x)
        
        # Output shape: (batch_size, seq_len, d_model)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Self-Attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        
        # Cross-Attention layer
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # Layer Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output):
        # x shape: (seq_len, batch_size, d_model)
        # enc_output shape: (enc_seq_len, batch_size, d_model)
        
        # Self-Attention
        hidden_states, _ = self.self_attn(x, x, x)
        x = self.norm1(x + hidden_states)
        
        # Cross-Attention
        hidden_states, _ = self.cross_attn(query=x, key=enc_output, value=enc_output)
        x = self.norm2(x + hidden_states)
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        # Output shape: (seq_len, batch_size, d_model)
        return x
    

class Decoder(nn.Module):
    def __init__(self, output_size, context_size, d_model, d_ff, num_heads, n_blocks):
        super().__init__()
        # Embedding layer to convert input tokens to vectors
        self.embedding = nn.Embedding(output_size, d_model)
        
        # Positional encoding to add position information
        self.pos_embedding = PositionalEncoding(context_size, d_model)
        
        # Stack of Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(n_blocks)
        ])

        # Output linear layer
        self.out = nn.Linear(d_model, output_size)

    def forward(self, x, enc_output):
        # x shape: (batch_size, seq_len)
        # enc_output shape: (enc_seq_len, batch_size, d_model)
        
        # Apply embedding and add positional encoding
        x = self.embedding(x) + self.pos_embedding(x)
        # x shape after embedding: (batch_size, seq_len, d_model)
        
        # Pass through each decoder block
        for block in self.blocks:
            x = block(x, enc_output)
        
        # Project to output size
        output = self.out(x)
        # output shape: (batch_size, seq_len, output_size)
        
        return output
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, d_ff, num_heads, n_blocks):
        super().__init__()
        
        # Encoder component
        self.encoder = Encoder(
            vocab_size,     # Size of the input vocabulary
            context_size,   # Maximum sequence length
            d_model,        # Dimensionality of the model
            d_ff,           # Dimensionality of the feedforward network
            num_heads,      # Number of attention heads
            n_blocks        # Number of encoder blocks
        )
        
        # Decoder component
        self.decoder = Decoder(
            vocab_size,     # Size of the output vocabulary (same as input in this case)
            context_size,   # Maximum sequence length
            d_model,        # Dimensionality of the model
            d_ff,           # Dimensionality of the feedforward network
            num_heads,      # Number of attention heads
            n_blocks        # Number of decoder blocks
        )

    def forward(self, input_encoder, input_decoder):
        # input_encoder shape: (batch_size, enc_seq_len)
        # input_decoder shape: (batch_size, dec_seq_len)
        
        # Pass input through the encoder
        enc_output = self.encoder(input_encoder)
        # enc_output shape: (batch_size, enc_seq_len, d_model)
        
        # Pass encoder output and decoder input through the decoder
        output = self.decoder(input_decoder, enc_output)
        # output shape: (batch_size, dec_seq_len, vocab_size)
        
        return output