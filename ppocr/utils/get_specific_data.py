import os

txt_full = '/mnt/ssd/martin/project/ocr/data/data_ocr_2/out_label_train.txt'
output = '/mnt/ssd/martin/project/ocr/data/data_ocr_2/out_label_train_card_only.txt'

def valid_data(row):
    for data_type in ['data_systhesis_idcard', 'real_idcard_data']:
        if data_type in row.split('\t')[0]:
            return True
    return False

with open(txt_full, 'r') as txt:
    data = txt.read().splitlines()

filter_data = [d + '\n' for d in data if valid_data(d)]
with open(output, 'w') as txt:
    txt.writelines(filter_data)
