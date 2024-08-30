from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)


def get_default(model_id, num_labels):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer