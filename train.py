import torch
import bentoml
import logging
import evaluate

from tqdm.auto import tqdm
from torch.optim import AdamW
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


logging.basicConfig(level='INFO')

DATA_PATH = './data/'
DATA_FILES = {
    'train': 'Train.csv',
    'val': 'Valid.csv',
    'test': 'Test.csv',
}

TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')


def tokenization(x):
    return TOKENIZER(x['text'], padding='max_length', truncation=True)


def prepare_datasets(data_path, data_files, lightweight=True):
    datasets = load_dataset(path=data_path, data_files=data_files)

    tokenized_datasets = datasets.map(tokenization, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    if lightweight: # reduce dataset size for faster training
        train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(50))
        val_dataset = tokenized_datasets['val'].shuffle(seed=42).select(range(10))
        test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(10))
    
    else:
        train_dataset = tokenized_datasets['train']
        val_dataset = tokenized_datasets['val']
        test_dataset = tokenized_datasets['test']

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    return train_dataloader, val_dataloader, test_dataloader


def train(train_dataloader):
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)  
    
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return device, model


def predict(device, model, test_dataloader):
    metric = evaluate.load('mae')

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])

    print(metric.compute())


def save_model(tokenizer, model):
    review_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    bentoml.transformers.save_model(
        name='review-classifier', 
        pipeline=review_classifier,
        signatures={ # enable Transformer to use adaptive batching at inference
            "__call__": {
                "batchable": True,
                "batch_dim": 0,
            },
        },
    )


if __name__=='__main__':
    logging.info('Preparing datasets')
    train_dataloader, val_dataloader, test_dataloader = prepare_datasets(DATA_PATH, DATA_FILES)

    logging.info('Training model')
    device, model = train(train_dataloader)

    logging.info('Testing model')
    predict(device, model, test_dataloader)

    logging.info('Saving model')
    save_model(TOKENIZER, model)
