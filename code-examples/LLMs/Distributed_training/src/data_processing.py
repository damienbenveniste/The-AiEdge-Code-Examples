

class DataProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _tokenize_function(self, examples):
        outputs = self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        return outputs

    def transform(self, data):
        tokenized_data = data.map(
            self._tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
        tokenized_data = tokenized_data.rename_column("label", "labels")
        tokenized_data.set_format(type='torch')
        return tokenized_data