import numpy as np
import pickle
import joblib
import pandas as pd
import string
import pdb
from NER_transformer import read_data, read_testdata
from scipy.special import softmax
from sklearn import preprocessing

illegal_boundary = "！？｡。＂＃＄％＆＇（）＊＋，，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.跟和同或及與、/個到……" + string.punctuation
num = '0123456789零一二三四五六七八九十百千'
letter = string.ascii_uppercase

def exchange_prob(predictions, exchange_threshold=0.5):
    # There may be a situation that the max confidence is "O" but some other entity also has high confidence.
    # Hence, if this entity has confidence higher than threshold, exchange it with the confidence of "O".
    predictions = np.array(predictions)

    mask = np.zeros(predictions.shape)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
        
            if max(predictions[i,j,:-1]) > exchange_threshold * predictions[i,j,-1] and np.argmax(predictions[i,j,:]) == len(tag2id)-1:
                predictions[i,j,np.argmax(predictions[i,j,:-1])], predictions[i,j,-1] = predictions[i,j,-1], predictions[i,j,np.argmax(predictions[i,j,:-1])]

            # get onehot, for voting
            mask[i,j,np.argmax(predictions[i,j,:])] = 1
    
    predictions = predictions * mask

    return predictions, mask


segment_length = 128

train_texts, train_tags = read_data('train_2_split.data')
tags = train_tags.copy()
unique_tags = set(tag for doc in tags for tag in doc)
unique_tags = sorted(unique_tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}   
id2tag = {id: tag for tag, id in tag2id.items()}
test_texts, test_tags = read_testdata('test.txt')

# change the probability file generated from NER_transformer.py
############################################ modification ############################################
with open('result_test_roberta_2000.pkl', 'rb') as f:   
    roberta_predictions = pickle.load(f)
    roberta_predictions, roberta_mask = exchange_prob(roberta_predictions)

with open('result_test_bert_1500.pkl', 'rb') as f:  
    bert_predictions = pickle.load(f)
    bert_predictions, bert_mask = exchange_prob(bert_predictions)

with open('result_test_macbert_2000.pkl', 'rb') as f:   
    macbert_predictions = pickle.load(f)
    macbert_predictions, macbert_mask = exchange_prob(macbert_predictions)

with open('result_test_electra_2000.pkl', 'rb') as f:
    electra_predictions = pickle.load(f)
    electra_predictions, electra_mask = exchange_prob(electra_predictions)

with open('result_test_roberta_wwm_ext_large_2000.pkl', 'rb') as f:
    roberta_wwm_ext_large_predictions = pickle.load(f)
    roberta_wwm_ext_large_predictions, roberta_wwm_ext_large_mask = exchange_prob(roberta_wwm_ext_large_predictions)

with open('result_test_longformer_2170.pkl', 'rb') as f:
    longformer_predictions = pickle.load(f)
    longformer_predictions, longformer_mask = exchange_prob(longformer_predictions)
########################################################################################################


############################################ modification ############################################
predictions = (np.array(bert_predictions) + np.array(electra_predictions) + np.array(roberta_predictions) + np.array(macbert_predictions) + np.array(roberta_wwm_ext_large_predictions) + np.array(longformer_predictions)) / 6 
########################################################################################################
predictions = softmax(predictions, axis=-1)

# for voting
############################################ modification ############################################
mask = roberta_mask + bert_mask + electra_mask + macbert_mask + roberta_wwm_ext_large_mask + longformer_mask 
######################################################################################################## 
mask = mask.astype('int64')
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if max(np.bincount(mask[i,j])) > 1 and max(np.bincount(mask[i,j])) < 10: # if there is two class get same votes
            mask[i,j] = predictions[i,j]


predictions_prob = predictions.copy()
predictions = np.argmax(mask, axis=2).tolist()

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
                if entity_text != '': # since one entity may be followed by another entity, ex. ......, O, O, B-location, I-location, B-person, I-person, O, O, ......
                    if label.split('-')[1] != entity_type: # if followed entity is different type
                        end_position = idx
                        if len(entity_text) > 1 and entity_text[0] not in illegal_boundary and entity_text[-1] not in illegal_boundary and (end_position - start_position == len(entity_text)):
                            entity_text_list.append(entity_text)
                            entity_type_list.append(entity_type)
                            start_position_list.append(start_position)
                            end_position_list.append(end_position)
                            article_id_list.append(article_id)
                        entity_text = char
                        entity_type = label.split('-')[1]
                        start_position = idx
                    else:
                        entity_text += char
                else:
                    entity_text += char
                    entity_type = label.split('-')[1]
                    start_position = idx
            elif label.split('-')[0] == 'I':
                if label.split('-')[1] == entity_type: # when entity is consistent with that of last character
                    entity_text += char
        else:
            if entity_text != '':
                end_position = idx
                if len(entity_text) > 1 and entity_text[0] not in illegal_boundary and entity_text[-1] not in illegal_boundary and (end_position - start_position == len(entity_text)): # filter out entity texts that only contain one char
                    entity_text_list.append(entity_text)
                    entity_type_list.append(entity_type)
                    start_position_list.append(start_position)
                    end_position_list.append(end_position)
                    article_id_list.append(article_id)
                elif len(entity_text) == 1 and entity_text == articles[article_id][idx-1][0] and entity_text in num:
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
final_df.to_csv('output_ensemble_test_check(bert_roberta_macbert_electra_roberta_wwm_ext_large_longformer)_hardvote_filter_e0.5_corrected3.tsv', index=False, sep='\t') 
######################################################################################################## 