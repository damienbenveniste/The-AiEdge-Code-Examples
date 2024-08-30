from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader
import torch
import evaluate
from huggingface_hub import login


HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"


class Trainer:

    def __init__(self, model, tokenizer, num_epochs, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accelerator = Accelerator()
        self.optimizer = optim.Adam(params=model.parameters())

    def train(self, tokenized_data):

        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data)

        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader
        )

        for epoch in range(self.num_epochs):
            model.train()
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                optimizer.step()

            eval_metric = self.eval(model, eval_dataloader)
            self.accelerator.print(f"epoch {epoch}:", eval_metric)

        self.save(model)

    def eval(self, model, eval_dataloader):
        model.eval()
        all_predictions = []
        all_labels = []

        accuracy_metric = evaluate.load("accuracy")

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            all_predictions.append(self.accelerator.gather(predictions))
            all_labels.append(self.accelerator.gather(batch["labels"]))

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        eval_metric = accuracy_metric.compute(
            predictions=all_predictions, 
            references=all_labels
        )

        return eval_metric

    def create_dataloaders(self, tokenized_data):

        train_dataloader = DataLoader(
            tokenized_data["train"], 
            shuffle=True, 
            batch_size=self.batch_size
        )
        eval_dataloader = DataLoader(
            tokenized_data["validation"], 
            shuffle=False, 
            batch_size=self.batch_size
        )
        return train_dataloader, eval_dataloader
    
    def save(self, model):
        unwrapped_model = self.accelerator.unwrap_model(model)
        repo_name = "my-distributed-model"
        login(token=HUGGINGFACE_TOKEN)
        unwrapped_model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)