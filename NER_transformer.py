from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, BertTokenizerFast, BertForTokenClassification,\
              ElectraTokenizerFast, ElectraForTokenClassification, XLNetTokenizer, XLNetForTokenClassification,\
              LongformerTokenizerFast, LongformerForTokenClassification, AlbertForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import re
import numpy as np
import torch
import pickle
import pdb
import torch.nn as nn 
import pandas as pd

# read character-based data, ex. train_2_split.data
def read_data(file_path):
    token_docs = []
    tag_docs = []
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        token_sentence = []
        tag_sentence = []
        for line in lines:
            line = line.decode()
            if len(line.split()) < 2:    
                token_docs.append(token_sentence)
                tag_docs.append(tag_sentence)
                token_sentence = []
                tag_sentence = []
            else:
                token, tag = line.split()
                token_sentence.append(token)
                tag_sentence.append(tag)

    return token_docs, tag_docs

# read the whole text, ex.test.txt
def read_testdata(file_path):
    token_docs = []
    tag_docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if (idx+1) % 5 == 2:
                token_sentence = []
                tag_sentence = []
                for cnt, char in enumerate(line):
                    token_sentence.append(char)
                    tag_sentence.append('O')
                    if (cnt+1) % segment_length == 0:
                        token_docs.append(token_sentence)
                        tag_docs.append(tag_sentence)
                        token_sentence = []
                        tag_sentence = []
                if len(token_sentence) > 0:
                    token_docs.append(token_sentence)
                    tag_docs.append(tag_sentence)

    return token_docs, tag_docs

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    # train or inference
    ############################################ modification ############################################
    is_train = True
    ########################################################################################################
    segment_length = 128

    ############################################ modification ############################################
    train_texts, train_tags = read_data('train_2_split.data')
    val_texts, val_tags = read_data('sample_split.data')
    ########################################################################################################

    tags = train_tags.copy()
    unique_tags = set(tag for doc in tags for tag in doc)
    unique_tags = sorted(unique_tags)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}   
    id2tag = {id: tag for tag, id in tag2id.items()}

    # select pretrained model 
    ############################################ modification ############################################
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # tokenizer = BertTokenizerFast.from_pretrained("clue/roberta_chinese_large")
    # tokenizer = ElectraTokenizerFast.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
    # tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    # tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-macbert-large')
    # tokenizer = BertTokenizerFast.from_pretrained("schen/longformer-chinese-base-4096")
    ########################################################################################################
    
    if is_train:
        # with tokenizerfast
        train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

        train_labels = encode_tags(train_tags, train_encodings)
        val_labels = encode_tags(val_tags, val_encodings)
        
        train_encodings.pop("offset_mapping") # we don't want to pass this to the model
        val_encodings.pop("offset_mapping")

        train_dataset = NERDataset(train_encodings, train_labels)
        val_dataset = NERDataset(val_encodings, val_labels)

        # select pretrained model 
        ############################################ modification ############################################
        model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('clue/roberta_chinese_large', num_labels=len(unique_tags))
        # model = ElectraForTokenClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('hfl/chinese-roberta-wwm-ext-large', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('hfl/chinese-macbert-large', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('schen/longformer-chinese-base-4096', num_labels=len(unique_tags))
        ########################################################################################################                            
                                                                                                                                          
        # train with trainer
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=10,             # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy='epoch',
            save_steps=217,
            learning_rate=1e-5
        )


        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )

        trainer.train()
    else:
        ############################################ modification ############################################
        test_texts, test_tags = read_testdata('data/test.txt')
        ########################################################################################################
        
        test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        test_labels = [[-100 for i in range(130)] for j in range(len(test_texts))]

        test_dataset = NERDataset(test_encodings, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # select pretrained model 
        ############################################ modification ############################################
        model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('clue/roberta_chinese_large', num_labels=len(unique_tags))
        # model = ElectraForTokenClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('hfl/chinese-roberta-wwm-ext-large', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('hfl/chinese-macbert-large', num_labels=len(unique_tags))
        # model = BertForTokenClassification.from_pretrained('schen/longformer-chinese-base-4096', num_labels=len(unique_tags))
        checkpoint = torch.load('bert_results/checkpoint-1500/pytorch_model.bin') # change checkpoint
        model.load_state_dict(checkpoint)
        model.to(device)
        ########################################################################################################

        # inference
        model.eval()
        predictions = []
        predictions_nomax = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            predictions += np.argmax(outputs.logits[:,1:-1,:].detach().cpu().numpy(), axis=2).tolist()
            predictions_nomax += outputs.logits[:,1:-1,:].detach().cpu().numpy().tolist()
        
        # save probabilities (necessary for ensemble) 
        ############################################ modification ############################################
        with open('/content/drive/My Drive/AIdea/transformer/result_test_longformer_lr1e-5_2170.pkl', 'wb') as f: # change filename
            pickle.dump(predictions_nomax, f)
        ########################################################################################################
        
        predictions_decode = []
        for prediction in predictions:
            predictions_decode.append([id2tag[item] for item in prediction])

        articles = []
        article = []
        for idx, sentence in enumerate(test_texts):
            article += [(sentence[i], predictions_decode[idx][i]) for i in range(len(sentence))]
            if len(sentence) != segment_length:
                articles.append(article)
                article = []
        
        entity_text_list = []
        entity_type_list = []
        start_position_list = []
        end_position_list = []
        article_id_list = []
        for article_id, article in enumerate(articles):
            entity_text = ''
            entity_type = ''
            for idx, item in enumerate(article):
                char = item[0]
                label = item[1]
                print(char + ' ' + label)
                if label != 'O':
                    if label.split('-')[0] == 'B':
                        entity_text += char
                        entity_type = label.split('-')[1]
                        start_position = idx
                    elif label.split('-')[0] == 'I':
                        if label.split('-')[1] == entity_type: # when entity is consistent with that of last character
                            entity_text += char
                        elif entity_type != '': # when entity is not consistent with that of last character
                            end_position = idx
                            entity_text_list.append(entity_text)
                            entity_type_list.append(entity_type)
                            start_position_list.append(start_position)
                            end_position_list.append(end_position)
                            article_id_list.append(article_id)
                            entity_text = char
                            entity_type = label.split('-')[1] 
                            start_position = idx
                        else: # when character become start of entity                         
                            entity_text = char
                            entity_type = label.split('-')[1]
                            start_position = idx
                else:
                    if entity_text != '':
                        end_position = idx
                        entity_text_list.append(entity_text)
                        entity_type_list.append(entity_type)
                        start_position_list.append(start_position)
                        end_position_list.append(end_position)
                        article_id_list.append(article_id)
                        entity_text = ''
                        entity_type = ''
        
        final_dict = {'article_id': article_id_list, 'start_position': start_position_list, 'end_position': end_position_list, 'entity_text': entity_text_list, 'entity_type': entity_type_list}
        final_df = pd.DataFrame.from_dict(final_dict)

        # save prediction
        ############################################ modification ############################################
        final_df.to_csv('/content/drive/My Drive/AIdea/transformer/output_test_longformer_lr1e-5_hf_0.tsv', index=False, sep='\t') # change output file's name