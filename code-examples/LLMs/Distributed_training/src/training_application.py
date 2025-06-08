"""
Main application script for distributed training of a text classification model.

This script demonstrates a complete pipeline for training a transformer model
on a classification task using distributed training with Hugging Face Accelerate.

The pipeline consists of:
1. Data loading from Hugging Face datasets
2. Model and tokenizer initialization
3. Data preprocessing and tokenization
4. Distributed training setup and execution

Usage:
    To run on a single GPU:
        python training_application.py
    
    To run on multiple GPUs:
        accelerate launch --multi_gpu training_application.py
    
    To run on multiple machines:
        accelerate launch --multi_gpu --num_machines 2 training_application.py
"""

from data_connector import DataConnector
from data_processing import DataProcessor
from training import Trainer
from model import get_default

# Configuration: Modify these parameters for your specific use case
MODEL_ID = 'gpt2'  # Pre-trained model to fine-tune
DATA_PATH = 'dair-ai/emotion'  # Dataset from Hugging Face Hub

def run():
    """
    Execute the complete distributed training pipeline.
    
    This function orchestrates the entire training process by:
    1. Loading the dataset
    2. Determining the number of classes automatically
    3. Loading and configuring the model and tokenizer
    4. Processing the data for training
    5. Running the distributed training
    
    The function handles all the coordination between different components
    and ensures proper setup for distributed training.
    """
    # Step 1: Load the dataset from Hugging Face Hub
    print(f"Loading dataset: {DATA_PATH}")
    data = DataConnector.get_data(DATA_PATH)
    
    # Step 2: Automatically determine the number of classes in the dataset
    # This works by examining unique labels in the training set
    num_labels = len(set(data['train']['label']))
    print(f"Detected {num_labels} classes in the dataset")
    
    # Step 3: Load pre-trained model and tokenizer
    # The model will be adapted for our specific number of classes
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = get_default(MODEL_ID, num_labels)

    # Step 4: Initialize data processor and trainer
    # Data processor handles tokenization and formatting
    data_processor = DataProcessor(tokenizer=tokenizer)
    
    # Trainer handles the distributed training loop
    # Adjust num_epochs and batch_size based on your dataset and hardware
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        num_epochs=3,  # Number of training epochs
        batch_size=16  # Batch size per device
    )
    
    # Step 5: Process the data for training
    print("Processing and tokenizing data...")
    tokenized_data = data_processor.transform(data)
    
    # Step 6: Start distributed training
    print("Starting distributed training...")
    trainer.train(tokenized_data)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    """
    Entry point for the training application.
    
    When running with accelerate launch, this script will automatically
    detect the distributed training configuration and set up the appropriate
    distributed training environment.
    """
    run()
