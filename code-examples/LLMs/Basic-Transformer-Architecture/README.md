# Basic Transformer Architecture

A complete, educational implementation of the Transformer architecture from scratch using PyTorch. This implementation closely follows the original "Attention Is All You Need" paper and provides clear, well-commented code for learning purposes.

## 📚 Overview

This module demonstrates the fundamental components of the Transformer architecture, including:

- **Attention Mechanisms** - Self-attention and multi-head attention
- **Positional Encoding** - Adding positional information to embeddings
- **Feed-Forward Networks** - Position-wise fully connected layers
- **Encoder-Decoder Architecture** - Complete transformer model

## 📁 Files

### `components.py`
Core building blocks of the transformer architecture:

#### Attention Classes
- **`Attention`** - Basic single-head attention mechanism
  - Implements Query, Key, Value transformations
  - Scaled dot-product attention
  - Input: `(batch_size, seq_len, d_in)`
  - Output: `(batch_size, seq_len, d_out)`

- **`BasicMultiheadAttention`** - Simple multi-head attention using multiple `Attention` instances
  - Creates separate attention heads
  - Concatenates outputs and applies linear projection

- **`MultiheadAttention`** - Efficient multi-head attention implementation
  - Single linear layer for Q, K, V projections
  - Parallel processing of multiple heads
  - More memory efficient than `BasicMultiheadAttention`

#### Positional Encoding Classes
- **`PositionalEncoding`** - Basic sinusoidal positional encoding
  - Uses sine and cosine functions
  - Fixed encoding for each position

- **`EfficientPositionalEncoding`** - Optimized version with pre-computed division terms
  - More efficient computation
  - Supports longer sequences (up to 5000 tokens by default)

#### Feed-Forward Network
- **`PositionwiseFeedForward`** - Two-layer MLP with ReLU activation
  - Expands to higher dimension (`d_ff`) then back to model dimension
  - Applied to each position independently

### `transformer.py`
Complete transformer model implementation:

#### Encoder Components
- **`EncoderBlock`** - Single encoder layer
  - Multi-head self-attention
  - Position-wise feed-forward network
  - Residual connections and layer normalization

- **`Encoder`** - Stack of encoder blocks
  - Embedding layer for input tokens
  - Positional encoding
  - Multiple encoder blocks

#### Decoder Components
- **`DecoderBlock`** - Single decoder layer
  - Masked multi-head self-attention
  - Multi-head cross-attention (encoder-decoder attention)
  - Position-wise feed-forward network
  - Residual connections and layer normalization

- **`Decoder`** - Stack of decoder blocks
  - Embedding layer for target tokens
  - Positional encoding
  - Multiple decoder blocks
  - Final linear projection to vocabulary

#### Complete Model
- **`Transformer`** - Full encoder-decoder transformer
  - Combines encoder and decoder
  - Handles input and target sequences

### `test_code.py`
Example usage and testing:

- Creates a simple vocabulary from sample text
- Initializes transformer model with specified hyperparameters
- Demonstrates forward pass through the model
- Shows next word prediction

## 🚀 Usage

### Basic Example

```python
import torch
from transformer import Transformer

# Model hyperparameters
VOCAB_SIZE = 100      # Size of vocabulary
CONTEXT_SIZE = 50     # Maximum sequence length
D_MODEL = 128         # Model dimension
D_FF = 512           # Feed-forward dimension
NUM_HEADS = 8        # Number of attention heads
N_BLOCKS = 6         # Number of encoder/decoder blocks

# Initialize model
model = Transformer(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    d_model=D_MODEL,
    d_ff=D_FF,
    num_heads=NUM_HEADS,
    n_blocks=N_BLOCKS
)

# Sample input (batch_size=2, seq_len=10)
encoder_input = torch.randint(0, VOCAB_SIZE, (2, 10))
decoder_input = torch.randint(0, VOCAB_SIZE, (2, 8))

# Forward pass
output = model(encoder_input, decoder_input)
print(f"Output shape: {output.shape}")  # (2, 8, VOCAB_SIZE)
```

### Running the Test Code

```bash
cd code-examples/LLMs/Basic-Transformer-Architecture
python test_code.py
```

Expected output:
```
Predicted next word: <some_word>
```

## 🔧 Architecture Details

### Attention Mechanism
```
Query = X @ W_Q
Key = X @ W_K  
Value = X @ W_V

Attention(Q,K,V) = softmax(QK^T / √d_k) @ V
```

### Multi-Head Attention
- Splits the model dimension across multiple heads
- Each head learns different types of relationships
- Outputs are concatenated and projected

### Positional Encoding
Uses sinusoidal functions to encode position:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Layer Structure
Each encoder/decoder block follows:
1. **Multi-Head Attention** (self-attention for encoder, masked for decoder)
2. **Add & Norm** (residual connection + layer normalization)
3. **Feed-Forward Network**
4. **Add & Norm** (residual connection + layer normalization)

Decoders additionally have cross-attention between steps 1 and 3.

## 📊 Model Parameters

With default settings from `test_code.py`:
- **Vocabulary Size**: Dynamic (based on input text)
- **Model Dimension**: 10
- **Feed-Forward Dimension**: 20
- **Attention Heads**: 2
- **Encoder/Decoder Blocks**: 10
- **Context Size**: 100

## 🎯 Learning Objectives

After studying this implementation, you should understand:

1. **Attention Mechanisms**: How queries, keys, and values work together
2. **Multi-Head Attention**: Parallel processing of different attention patterns
3. **Positional Encoding**: How transformers handle sequence order
4. **Encoder-Decoder Architecture**: How information flows through the model
5. **Residual Connections**: Why they're crucial for training deep networks
6. **Layer Normalization**: Stabilizing training in transformer models

## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## 💡 Extensions

Try these modifications to deepen your understanding:

1. **Add dropout** for regularization
2. **Implement causal masking** for autoregressive generation
3. **Add different positional encodings** (learned, rotary, etc.)
4. **Experiment with different activation functions** (GELU, Swish)
5. **Add pre-layer normalization** (Pre-LN vs Post-LN)
6. **Implement gradient checkpointing** for memory efficiency

## ⚠️ Notes

- This implementation prioritizes clarity over efficiency
- For production use, consider using `torch.nn.Transformer` or Hugging Face Transformers
- The model uses PyTorch's `MultiheadAttention` in `transformer.py` for the encoder/decoder blocks
- Custom attention implementations in `components.py` are for educational purposes