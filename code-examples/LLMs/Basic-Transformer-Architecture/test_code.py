import torch
from transformer import Transformer

# Define special tokens
SOS_token = 0  # Start of sequence token
EOS_token = 1  # End of sequence token
PAD_token = 2  # Padding token

# Initialize vocabulary with special tokens
index2words = {
    SOS_token: 'SOS',
    EOS_token: 'EOS',
    PAD_token: 'PAD'
}

# Sample sentence to build vocabulary
words = "How are you doing ? I am good and you ?"
words_list = set(words.lower().split(' '))
for word in words_list:
    index2words[len(index2words)] = word
    
# Create reverse mapping: word to index
words2index = {w: i for i, w in index2words.items()}

def convert2tensors(sentence, max_len):
    """Convert a sentence to a padded tensor of word indices."""
    words_list = sentence.lower().split(' ')
    padding = ['PAD'] * (max_len - len(words_list))
    words_list.extend(padding)
    indexes = [words2index[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

# Set model hyperparameters
D_MODEL = 10
VOCAB_SIZE = len(words2index)
N_BLOCKS = 10
D_FF = 20
CONTEXT_SIZE = 100
NUM_HEADS = 2

# Initialize the Transformer model
transformer = Transformer(
    vocab_size=VOCAB_SIZE, 
    context_size=CONTEXT_SIZE, 
    d_model=D_MODEL, 
    d_ff=D_FF, 
    num_heads=NUM_HEADS, 
    n_blocks=N_BLOCKS
)

# Prepare input sentences
input_sentence = "How are you doing ?"
output_sentence = "I am good and"

# Convert sentences to tensors
input_encoder = convert2tensors(input_sentence, CONTEXT_SIZE)
input_decoder = convert2tensors(output_sentence, CONTEXT_SIZE)

# Run the model
output = transformer(input_encoder, input_decoder)

# Get the most likely next word
_, indexes = output.squeeze().topk(1)
predicted_word = index2words[indexes[3].item()]
print(f"Predicted next word: {predicted_word}")
# > 'are' (for example)
