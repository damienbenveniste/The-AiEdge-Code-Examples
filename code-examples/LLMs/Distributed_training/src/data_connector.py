from datasets import load_dataset

class DataConnector:
    """
    A utility class for loading datasets from Hugging Face datasets library.
    
    This class provides a simple interface to load datasets that can be used
    for training machine learning models. It acts as an abstraction layer
    between the application and the datasets library.
    """

    @staticmethod
    def get_data(path):
        """
        Load a dataset from the specified path.
        
        Args:
            path (str): The path or identifier of the dataset to load.
                       This can be a dataset name from Hugging Face Hub
                       (e.g., 'dair-ai/emotion') or a local path.
        
        Returns:
            datasets.Dataset: The loaded dataset object containing train/validation/test splits.
                             The exact structure depends on the specific dataset loaded.
        
        Example:
            >>> data = DataConnector.get_data('dair-ai/emotion')
            >>> print(data['train'][0])  # First training example
        """
        return load_dataset(path)