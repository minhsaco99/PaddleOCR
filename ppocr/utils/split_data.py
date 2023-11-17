import os
import random
random.seed(5699)
src_file = '/mnt/ssd/martin/project/ocr/data/data_ocr_2/out_label.txt'
split_rate = 0.8

with open(src_file, 'r') as txt:
    data = txt.read().splitlines()

random.shuffle(data)
train_list = data[:int(len(data)*0.8)]
val_list = data[int(len(data)*0.8):]

with open("/mnt/ssd/martin/project/ocr/data/data_ocr_2/out_label_train.txt", 'w') as txt:
    for line in train_list:
        txt.write(line + '\n')

with open("/mnt/ssd/martin/project/ocr/data/data_ocr_2/out_label_val.txt", 'w') as txt:
    for line in val_list:
        txt.write(line + '\n')