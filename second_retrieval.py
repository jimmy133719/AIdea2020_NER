import numpy as np
import pandas as pd
import re
import string
import pickle
import joblib
import monpa
import json
import pdb
from NER_transformer import read_data, read_testdata

# load lookup dictionary
with open('post_process_dict.json', 'r', encoding='utf-8') as f:
    post_process_dict = json.load(f)
surname_list = post_process_dict['surname']

def chunks(text, n):
    """Yield successive n-sized chunks from lst."""
    text_split = []
    for i in range(0, len(text), n):
        text_split.append(text[i:i + n])
    return text_split

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

num = '0123456789'
chinese_num = '一二三四五六七八九零十百千'
letter = string.ascii_uppercase
f_unit = post_process_dict['f_unit']
b_unit = post_process_dict['b_unit']
conj = post_process_dict['conj']
family_list = post_process_dict['family_list']
time_list = post_process_dict['retrieve_time_list'] 
surname_back_list = post_process_dict['surname_back_list']
casuality_list = post_process_dict['casuality_list']

illegal_text = post_process_dict['delete_time_list'] + post_process_dict['medical_brand'] + post_process_dict['local_brand'] + ['台灣', '臺灣']
segment_length = 128

############################################ modification ############################################
df_old = pd.read_csv('output_ensemble_test_check(bert_roberta_macbert_electra_roberta_wwm_ext_large_longformer)_hardvote_filter_e0.5_corrected3.tsv', sep='\t')
df_new = df_old.copy()
######################################################################################################## 

# load test data
############################################ modification ############################################
all_texts = []
with open('test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if (idx+1) % 5 == 2:
            all_texts.append(line)
######################################################################################################## 

id_cnt = 0
time_cnt = 0
unit_cnt = 0
decimal_cnt = 0
decimal_text = ''
decimal_list = []
entity_text = ''
entity_text2 = ''

# load sequence classification prediction
contain_entity = joblib.load('seq_contain_entity.pkl')

for text_id, text in enumerate(all_texts):
    
    text_split = chunks(text, 128)
    for chunk_id, chunk in enumerate(text_split):
        result_pseg = monpa.pseg(chunk) # use monpa package to do word segmentation
        current_id = 128 * chunk_id
        for pseg_id, pseg in enumerate(result_pseg):
            if  pseg[0] in time_list and contain_entity[text_id][current_id] == 1: # retrieve time entity which is inside the sequence predicted to contain entity
                df_new = df_new.append({'article_id': text_id, 'start_position': current_id, 'end_position': current_id+len(pseg[0]), 'entity_text': pseg[0], 'entity_type':'time'}, ignore_index=True)
            elif pseg_id < len(result_pseg) - 1 and (pseg[0] + result_pseg[pseg_id+1][0]) in time_list and contain_entity[text_id][current_id] == 1: # retrieve time entity which is inside the sequence predicted to contain entity
                df_new = df_new.append({'article_id': text_id, 'start_position': current_id, 'end_position': current_id+len(pseg[0])+1, 'entity_text': pseg[0]+result_pseg[pseg_id+1][0], 'entity_type':'time'}, ignore_index=True)
            elif len(pseg[0]) > 1 and (pseg[0] in family_list): # retrieve family entity
                df_new = df_new.append({'article_id': text_id, 'start_position': current_id, 'end_position': current_id+len(pseg[0]), 'entity_text': pseg[0], 'entity_type':'family'}, ignore_index=True)
                print(pseg)
            elif pseg_id > 0 and result_pseg[pseg_id-1][0] in surname_list and pseg[0] in surname_back_list: # retreive name entity with surname + title
                df_new = df_new.append({'article_id': text_id, 'start_position': current_id-1, 'end_position': current_id+len(pseg[0]), 'entity_text': result_pseg[pseg_id-1][0] + pseg[0], 'entity_type':'name'}, ignore_index=True)
                print(result_pseg[pseg_id-1][0] + pseg[0])
            current_id += len(pseg[0])

df_new = df_new.drop_duplicates(subset=['article_id','start_position'])
df_new = df_new.drop_duplicates(subset=['article_id','end_position'])

for text_id, text in enumerate(all_texts):
    df_temp = df_new.loc[df_new.article_id==text_id]
    for index, row in df_temp.iterrows():
        isChinesenum = ''.join([item for item in row.entity_text if item in chinese_num]) == row.entity_text
        if (isChinesenum or isEnglish(row.entity_text)) and (text[row.start_position-1] in f_unit or text[row.end_position] in b_unit): # drop the prediction that contains number/Chinese number/English char and is incomplete
            eleminate_index = row.name
            df_new = df_new.drop(index=eleminate_index) 
            print(row.entity_text + '--------->' + text[row.start_position-2:row.end_position+2])
        elif isEnglish(row.entity_text) and (isEnglish(text[row.start_position-1]) or isEnglish(text[row.end_position])): # drop the prediction that contains number/English char, and its surroundings are also number/Chinese number/English char 
            eleminate_index = row.name
            df_new = df_new.drop(index=eleminate_index)  
            print(row.entity_text + '--------->' + text[row.start_position-2:row.end_position+2])
        elif isChinesenum and (text[row.start_position-1] in chinese_num or text[row.end_position] in chinese_num): # drop the prediction that contains Chinese number char, and its surroundings are also number/Chinese number/English char
            eleminate_index = row.name
            df_new = df_new.drop(index=eleminate_index) 
            print(row.entity_text + '--------->' + text[row.start_position-2:row.end_position+2])
        elif (row.entity_text[0] in b_unit and text[row.start_position-1] in chinese_num): # drop the prediction whose followed char is number and is part of entity
            eleminate_index = row.name
            df_new = df_new.drop(index=eleminate_index) 
            print(row.entity_text + '--------->' + text[row.start_position-2:row.end_position+2])
        elif row.entity_text in illegal_text:
            if row.entity_text in post_process_dict['local_brand']: # convert the label of local brand to profession, instead of location
                df_new.loc[row.name, 'entity_type'] = 'profession'
            else: # delete other illegal prediction
                eleminate_index = row.name
                df_new = df_new.drop(index=eleminate_index) 
        
        # retrieve two entities that are actually together
        next_row_index = row.name + 1
        try:
            next_row = df_new.loc[next_row_index]
            if row.article_id == next_row.article_id and row.end_position == next_row.start_position and row.entity_type == next_row.entity_type:
                df_new = df_new.append({'article_id': row.article_id, 'start_position': row.start_position, 'end_position': next_row.end_position, 'entity_text': row.entity_text + next_row.entity_text, 'entity_type':row.entity_type}, ignore_index=True)
                df_new = df_new.drop(index=row.name)
                df_new = df_new.drop(index=next_row_index)
        except:
            print('key error')


for text_id, text in enumerate(all_texts):
    for char_id, char in enumerate(text):
        # retrieve conj
        if char in conj:
            f_entity = df_new[(df_new.article_id==text_id) & (df_new.end_position==char_id)]
            b_entity = df_new[(df_new.article_id==text_id) & (df_new.start_position==char_id+1)]
            if len(f_entity) > 0 and len(b_entity) > 0 and f_entity.entity_type.item() == b_entity.entity_type.item():
                new_start_position = f_entity.start_position.item()
                new_end_position = b_entity.end_position.item()
                new_entity_text = f_entity.entity_text.item() + char + b_entity.entity_text.item()
                new_entity_type = f_entity.entity_type.item()
                eleminate_index1 = int(df_new[(df_new.entity_text == f_entity.entity_text.item()) & (df_new.article_id == f_entity.article_id.item()) & (df_new.start_position == f_entity.start_position.item())].index[0])
                eleminate_index2 = int(df_new[(df_new.entity_text == b_entity.entity_text.item()) & (df_new.article_id == b_entity.article_id.item()) & (df_new.start_position == b_entity.start_position.item())].index[0])
                df_new = df_new.drop(index=eleminate_index1)
                df_new = df_new.drop(index=eleminate_index2)
                df_new = df_new.append({'article_id': text_id, 'start_position': new_start_position, 'end_position': new_end_position, 'entity_text': new_entity_text, 'entity_type':new_entity_type}, ignore_index=True)

        # retrieve id
        if char in letter:
            entity_text = char
            start_position = char_id
            id_cnt = 1
        elif char in num and id_cnt > 0:
            entity_text += char
            id_cnt += 1
        else:
            if id_cnt >= 9:
                end_position = char_id
                modified_df = df_new[(df_new.article_id==text_id) & (df_new.start_position>=start_position) & (df_new.end_position<=end_position)]
                if len(modified_df) > 0:
                    df_new = df_new[df_new.entity_text != modified_df['entity_text'].item()]
                df_new = df_new.append({'article_id':text_id, 'start_position':start_position, 'end_position': end_position, 'entity_text': entity_text, 'entity_type':'ID'}, ignore_index=True)
            entity_text = ''
            id_cnt = 0
   
        # retrieve clinical event
        for item in casuality_list:
            if item == text[char_id:char_id + len(item)]:
                clinical_entity_text = item
                clinical_start_position = char_id
                clinical_end_position = char_id + len(item)
                df_new = df_new.append({'article_id':text_id, 'start_position':clinical_start_position, 'end_position': clinical_end_position, 'entity_text': clinical_entity_text, 'entity_type':'clinical_event'}, ignore_index=True)
                break



df_new = df_new.sort_values(by=['article_id', 'start_position'])

# save prediction
############################################ modification ############################################
df_new.to_csv('output_ensemble_test_check_rule2(bert_roberta_macbert_electra_roberta_wwm_ext_large_longformer)_hardvote_filter_e0.5_corrected5.tsv', index=False, sep='\t') 
########################################################################################################
