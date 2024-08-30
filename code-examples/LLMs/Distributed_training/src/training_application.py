from data_connector import DataConnector
from data_processing import DataProcessor
from training import Trainer
from model import get_default

MODEL_ID = 'gpt2'
DATA_PATH = 'dair-ai/emotion'

def run():

    data = DataConnector.get_data(DATA_PATH)
    num_labels = len(set(data['train']['label']))
    model, tokenizer = get_default(MODEL_ID, num_labels)

    data_processor = DataProcessor(tokenizer=tokenizer)
    trainer = Trainer(model, tokenizer, num_epochs=3, batch_size=16)
    tokenized_data = data_processor.transform(data)
    trainer.train(tokenized_data)

if __name__ == '__main__':
    run()
