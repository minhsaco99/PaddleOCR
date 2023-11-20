LD_LIBRARY_PATH=~/anaconda3/envs/ocr/lib/ python tools/train.py \
                                                -c=configs/rec/rec_svtrnet_cppd_base_en.yml \
                                                -o Global.pretrained_model=models/rec_svtr_cppd_base_48_160_en_train/best_model
                                            