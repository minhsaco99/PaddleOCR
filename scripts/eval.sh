LD_LIBRARY_PATH=~/anaconda3/envs/ocr/lib/ python tools/eval.py \
                                                -c=configs/rec/rec_svtrnet_cppd_base_en.yml \
                                                -o Global.checkpoints=models/techainer_svtr_cppd_base_fulltext/best_accuracy
                                            