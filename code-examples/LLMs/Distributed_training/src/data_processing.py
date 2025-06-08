

class DataProcessor:
    """
    A class responsible for processing and tokenizing text data for model training.
    
    This class handles the transformation of raw text data into tokenized format
    that can be consumed by transformer models. It performs tokenization, padding,
    truncation, and format conversion for efficient training.
    """

    def __init__(self, tokenizer):
        """
        Initialize the DataProcessor with a tokenizer.
        
        Args:
            tokenizer: A Hugging Face tokenizer instance (e.g., AutoTokenizer)
                      used to convert text into tokens that the model can understand.
        """
        self.tokenizer = tokenizer

    def _tokenize_function(self, examples):
        """
        Tokenize a batch of text examples.
        
        This is a private method that applies tokenization to text data with
        specific parameters for consistent input formatting.
        
        Args:
            examples (dict): A batch of examples containing "text" field with
                           string data to be tokenized.
        
        Returns:
            dict: Tokenized outputs containing input_ids, attention_mask, etc.
                 The exact keys depend on the tokenizer used.
        """
        outputs = self.tokenizer(
            examples["text"],  # The text data to tokenize
            truncation=True,   # Truncate sequences longer than max_length
            padding="max_length",  # Pad shorter sequences to max_length
            max_length=128     # Maximum sequence length (adjust based on your needs)
        )
        return outputs

    def transform(self, data):
        """
        Transform the entire dataset by applying tokenization and formatting.
        
        This method processes the dataset by:
        1. Tokenizing all text data in batches for efficiency
        2. Removing the original text column (no longer needed)
        3. Renaming 'label' to 'labels' (required by Hugging Face models)
        4. Converting to PyTorch format for training
        
        Args:
            data (datasets.Dataset): The raw dataset containing text and labels.
        
        Returns:
            datasets.Dataset: The processed dataset ready for model training,
                             with tokenized inputs and proper formatting.
        """
        # Apply tokenization to all examples in batches for efficiency
        tokenized_data = data.map(
            self._tokenize_function, 
            batched=True,  # Process multiple examples at once for speed
            remove_columns=["text"]  # Remove original text column to save memory
        )
        
        # Rename 'label' to 'labels' - this is the expected input name for HF models
        tokenized_data = tokenized_data.rename_column("label", "labels")
        
        # Set format to PyTorch tensors for compatibility with PyTorch training
        tokenized_data.set_format(type='torch')
        
        return tokenized_data