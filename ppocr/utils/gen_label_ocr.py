import os
from pathlib import Path, PosixPath
import re
import unicodedata
import argparse
from typing import List, Dict
from threading import Thread, Lock

lock = Lock()

def read_file_and_split(file: PosixPath):
    if file.is_file():
        return file.open().read().splitlines()
    else:
        raise FileNotFoundError("File not found")
    
def gen_rec(folder: PosixPath, vocab: List[str], output_dict: Dict[str, str]):
    src_path = folder / 'src.txt'
    tgt_path = folder / 'tgt.txt'
    src_images = read_file_and_split(src_path)
    label_lists = read_file_and_split(tgt_path)
    for img_name, label in zip(src_images, label_lists):
        img_path = folder / "images" / img_name
        clean_label = ""
        for char in [*label]:
            if char in vocab:
                clean_label += char
        clean_label = clean_label.replace(';', ' ')
        with lock:
            output_dict[img_path] = clean_label

def gen_rec_label(input_path: PosixPath, output_label: PosixPath, vocab: List[str]):
    output_dict = dict()
    thread_list = []

    for folder in input_path.iterdir():
        if "data_systhesis_idcard" in str(folder):
            for sub_folder in folder.iterdir():
                if sub_folder.is_dir():
                    thread = Thread(target=gen_rec, args=(sub_folder, vocab, output_dict, ))
                    thread_list.append(thread)
                    thread.start()

        # elif folder.is_dir():
        elif folder.is_dir() and "real_idcard_data" in str(folder):
            thread = Thread(target=gen_rec, args=(folder, vocab, output_dict, ))
            thread_list.append(thread)
            thread.start()

    for thr in thread_list:
        thr.join()

    with open(output_label, 'w') as txt:
        for img_path, label in output_dict.items():
            txt.write(f"{img_path}\t{label}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        default=".",
        help='Input_label or input path to be converted')
    parser.add_argument(
        '--vocab_path',
        type=str,
        required=True,
        help='Input_label or input path to be converted')
    parser.add_argument(
        '--output_label',
        type=str,
        default="out_label.txt",
        help='Output file name')

    args = parser.parse_args()

    vocab = read_file_and_split(Path(args.vocab_path))
    print("Generate rec label")
    gen_rec_label(Path(args.input_path), Path(args.input_path) / args.output_label, vocab)
