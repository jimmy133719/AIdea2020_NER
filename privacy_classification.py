import pdb
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification, AdamW, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from NER_transformer import read_data, read_testdata, encode_tags

class AIdeaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    # train or inference
    ############################################ modification ############################################
    is_train = True
    ########################################################################################################
    segment_length = 128

    train_texts, train_tags = read_data('train_2_split.data')
    val_texts, val_tags = read_data('sample_split.data')

    tags = train_tags.copy()
    unique_tags = set(tag for doc in tags for tag in doc)
    unique_tags = sorted(unique_tags)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}   
    id2tag = {id: tag for tag, id in tag2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    if is_train:
        # with tokenizerfast
        train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

        train_labels = encode_tags(train_tags, train_encodings)
        val_labels = encode_tags(val_tags, val_encodings)
        
        train_encodings.pop("offset_mapping") # we don't want to pass this to the model
        val_encodings.pop("offset_mapping")
        train_encodings.pop("token_type_ids")
        val_encodings.pop("token_type_ids")

        train_labels = [1 if len(set(item))>2 else 0 for item in train_labels]
        val_labels = [1 if len(set(item))>2 else 0 for item in val_labels]


        # convert into Dataset
        train_dataset = AIdeaDataset(train_encodings, train_labels)
        val_dataset = AIdeaDataset(val_encodings, val_labels)


        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # train with trainer
        training_args = TrainingArguments(
            output_dir='./bertseq_sentence_results',          # output directory
            num_train_epochs=5,              # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training 
            per_device_eval_batch_size=32,   # batch size for evaluation 
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./bertseq_sentence_logs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy='epoch',
            # save_steps=500,
        )

        # select pretrained model 
        ############################################ modification ############################################
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
        #######################################################################################################

        
        trainer = Trainer(
            model=model,                         # the instantiated ?? Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )

        trainer.train()
 
    else:
        test_texts, test_tags = read_testdata('test.txt')

        
        test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        test_labels = [[-100 for i in range(130)] for j in range(len(test_texts))]

        test_encodings.pop("token_type_ids")
        test_labels = [1 if len(set(item))>2 else 0 for item in test_labels]


        test_dataset = AIdeaDataset(test_encodings, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # select pretrained model 
        ############################################ modification ############################################
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
        checkpoint = torch.load('bertseq_sentence_results/checkpoint-7500/pytorch_model.bin') # change checkpoint
        model.load_state_dict(checkpoint)
        model.to(device)
        #######################################################################################################

        # inference
        model.eval()
        predictions = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            outputs_softmax = outputs.logits.detach().cpu().numpy()
            outputs_softmax = softmax(outputs_softmax, axis=-1)
            predictions += [1 if item[1] > 0.5 else item[1] for item in outputs_softmax]

        # save probability output of sequence classification model 
        ############################################ modification ############################################
        joblib.dump(predictions, 'bertseq_sentence_predictions.pkl')
        #######################################################################################################

