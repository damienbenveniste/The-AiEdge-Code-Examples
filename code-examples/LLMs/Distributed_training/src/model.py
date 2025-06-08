from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)


def get_default(model_id, num_labels):
    """
    Load and configure a pre-trained model and tokenizer for sequence classification.
    
    This function sets up a complete model-tokenizer pair ready for fine-tuning
    on classification tasks. It handles the necessary configuration to ensure
    compatibility between the model and tokenizer.
    
    Args:
        model_id (str): The identifier of the pre-trained model to load.
                       Examples: 'gpt2', 'bert-base-uncased', 'distilbert-base-uncased'
        num_labels (int): The number of classes in your classification task.
                         This determines the output dimension of the classifier head.
    
    Returns:
        tuple: A tuple containing:
            - model: AutoModelForSequenceClassification instance ready for training
            - tokenizer: AutoTokenizer instance configured to work with the model
    
    Example:
        >>> model, tokenizer = get_default('gpt2', num_labels=6)
        >>> # Now ready for training on a 6-class classification problem
    """
    # Load the pre-trained model with a classification head
    # ignore_mismatched_sizes=True allows loading models even if the classifier
    # head size doesn't match (which is expected when changing num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Ignore size mismatch in classifier head
    )
    
    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure padding token - some models (like GPT-2) don't have a pad token
    # We use the end-of-sequence token as the padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure the model knows about the padding token ID
    # This is important for proper attention masking during training
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer