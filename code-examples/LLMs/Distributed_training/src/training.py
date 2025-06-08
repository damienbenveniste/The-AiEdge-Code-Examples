from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader
import torch
import evaluate
from huggingface_hub import login

# Configuration: Replace with your actual Hugging Face token for model upload
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"


class Trainer:
    """
    A distributed training class that leverages Hugging Face Accelerate for multi-GPU training.
    
    This class handles the complete training pipeline including:
    - Distributed training setup using Accelerate
    - Training and evaluation loops
    - Model saving and uploading to Hugging Face Hub
    
    The Accelerator automatically handles device placement, gradient accumulation,
    and distributed training across multiple GPUs or machines.
    """

    def __init__(self, model, tokenizer, num_epochs, batch_size):
        """
        Initialize the distributed trainer.
        
        Args:
            model: The PyTorch model to train (e.g., AutoModelForSequenceClassification)
            tokenizer: The tokenizer associated with the model
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training and evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Accelerator handles distributed training, mixed precision, gradient accumulation
        # It automatically detects the available hardware and configures accordingly
        self.accelerator = Accelerator()
        
        # Initialize optimizer - Adam is a good default for most transformer models
        self.optimizer = optim.Adam(params=model.parameters())

    def train(self, tokenized_data):
        """
        Execute the complete training pipeline.
        
        This method orchestrates the entire training process:
        1. Creates data loaders for training and evaluation
        2. Prepares model, optimizer, and data loaders for distributed training
        3. Runs the training loop with periodic evaluation
        4. Saves the final model
        
        Args:
            tokenized_data (datasets.Dataset): The preprocessed dataset containing
                                             train and validation splits
        """
        # Create PyTorch DataLoaders from the tokenized dataset
        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data)

        # Prepare everything for distributed training
        # This handles device placement, distributed wrappers, etc.
        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader
        )

        # Main training loop
        for epoch in range(self.num_epochs):
            # Set model to training mode (enables dropout, batch norm updates, etc.)
            model.train()
            
            # Iterate through training batches
            for batch in train_dataloader:
                # Clear gradients from previous iteration
                self.optimizer.zero_grad()
                
                # Forward pass: get model outputs (logits, loss, etc.)
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass: compute gradients
                # accelerator.backward() handles distributed gradient computation
                self.accelerator.backward(loss)
                
                # Update model parameters
                optimizer.step()

            # Evaluate model performance at the end of each epoch
            eval_metric = self.eval(model, eval_dataloader)
            
            # Print results (only on main process in distributed setting)
            self.accelerator.print(f"epoch {epoch}:", eval_metric)

        # Save the trained model
        self.save(model)

    def eval(self, model, eval_dataloader):
        """
        Evaluate the model on the validation set.
        
        This method runs inference on the evaluation dataset and computes
        accuracy metrics. It handles distributed evaluation by gathering
        predictions from all processes.
        
        Args:
            model: The model to evaluate
            eval_dataloader: DataLoader containing validation data
            
        Returns:
            dict: Evaluation metrics (e.g., {'accuracy': 0.85})
        """
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        all_predictions = []
        all_labels = []

        # Load the accuracy metric
        accuracy_metric = evaluate.load("accuracy")

        # Iterate through evaluation batches
        for batch in eval_dataloader:
            # Disable gradient computation for efficiency
            with torch.no_grad():
                outputs = model(**batch)
                
            # Get predicted class (highest logit value)
            predictions = outputs.logits.argmax(dim=-1)

            # Gather predictions and labels from all distributed processes
            # This is crucial for correct evaluation in multi-GPU setups
            all_predictions.append(self.accelerator.gather(predictions))
            all_labels.append(self.accelerator.gather(batch["labels"]))

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # Compute the final accuracy metric
        eval_metric = accuracy_metric.compute(
            predictions=all_predictions, 
            references=all_labels
        )

        return eval_metric

    def create_dataloaders(self, tokenized_data):
        """
        Create PyTorch DataLoaders for training and evaluation.
        
        DataLoaders handle batching, shuffling, and efficient data loading
        during training. Training data is shuffled for better convergence,
        while evaluation data is not shuffled for reproducible results.
        
        Args:
            tokenized_data (datasets.Dataset): The tokenized dataset with
                                             train and validation splits
                                             
        Returns:
            tuple: (train_dataloader, eval_dataloader)
        """
        # Training DataLoader with shuffling for better gradient estimates
        train_dataloader = DataLoader(
            tokenized_data["train"], 
            shuffle=True,  # Shuffle training data each epoch
            batch_size=self.batch_size
        )
        
        # Evaluation DataLoader without shuffling for consistent results
        eval_dataloader = DataLoader(
            tokenized_data["validation"], 
            shuffle=False,  # Don't shuffle validation data
            batch_size=self.batch_size
        )
        
        return train_dataloader, eval_dataloader
    
    def save(self, model):
        """
        Save and upload the trained model to Hugging Face Hub.
        
        This method unwraps the model from any distributed training wrappers,
        then uploads both the model and tokenizer to the Hugging Face Hub
        for easy sharing and deployment.
        
        Args:
            model: The trained model (potentially wrapped by Accelerator)
        """
        # Unwrap the model from distributed training wrappers
        # This is necessary to get the original model for saving
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # Set the repository name for the Hub
        repo_name = "my-distributed-model"
        
        # Authenticate with Hugging Face Hub
        login(token=HUGGINGFACE_TOKEN)
        
        # Upload model and tokenizer to the Hub
        # This makes the model publicly available for download and use
        unwrapped_model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)