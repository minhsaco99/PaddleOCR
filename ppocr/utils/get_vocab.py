import os
from pathlib import Path
import re
import unicodedata

def remove_stupid_char(str):
    str = re.sub("\x08.", "", str)
    map_character = {
        "ş": "s",
        "": " ",
        "❶": "",
        "“": "\"",
        "ç": "c",
        "ﬃ": "ffi",
        "ﬀ": "ff",
        "▬": "-",
        "": " ",
        "→": "->",
        "Ï": "Ĩ",
        "Ä": "Ã",
        "": " ",
        "Ą": "A",
        "”": "\"",
        "ø": "ơ",
        "–": "-",
        "Å": "Â",
        "Ö": "Õ",
        "■": "",
        "": " ",
        "": "",
        "ƣ": "",
        "−": "-",
        "": "",
        "̣": "",
        "—": "-",
        "": "",
        "Ź": "Z",
        "Ş": "S",
        "년": "",
        "Ø": "0",
        "ö": "õ",
        "": "",
        "°": "o",
        "ˆ": "^",
        "": "",
        "": " ",
        "Ť": "",
        "Ð": "Đ",
        "Ć": "C",
        "Ƣ": "",
        "�": "",
        "ﬂ": "fl",
        "ï": "ĩ",
        "": "",
        "Ñ": "N",
        "•": ".",
        "×": "x",
        "ä": "ã",
        "❷": "",
        "´": "\"",
        "ﬁ": "fi",
        "©": "",
        "❹": "",
        "": "",
        "": "",
        "ñ": "n",
        "’": "\"",
        "å": "ả",
        "●": ".",
        "­": "",
        "": "",
        "‘": "\"",
        "≤": "<=",
        "": " ",
        "‐": "-",
        "š": "s",
        "Ł": "L",
        "…": "...",
        "≥": ">=",
        "": "",
        "𝑎": "a"
    }
    str = ''.join(map(lambda c: map_character.get(c, c), str))
    return str

def get_vocab(folder_path: str):
    with open(os.path.join(folder_path, 'tgt.txt')) as d:
        data = d.read()
    data = data.replace(' ', '')
    data = data.replace('\;', ' ')
    data = data.replace('\n', '')
    data = remove_stupid_char(data)
    data = data.replace(' ', '')
    return set([*data])

def can_display_unicode(character):
    try:
        unicodedata.name(character)
        return True
    except ValueError:
        return False
    
if __name__ == "__main__":
    final_vocab = set()
    vocab_folder = Path("/mnt/ssd/martin/project/ocr/data/data_ocr_2")
    output_vocab = vocab_folder / "vocab_idcard.txt"

    for folder in vocab_folder.iterdir():
        if "data_systhesis_idcard" in str(folder):
            for sub_folder in folder.iterdir():
                if sub_folder.is_dir():
                    final_vocab.update(get_vocab(sub_folder))
        elif folder.is_dir() and "real_idcard_data" in str(folder):
            final_vocab.update(get_vocab(folder))

    final_vocab_clean = set()

    for char in final_vocab:
        if can_display_unicode(char):
            final_vocab_clean.add(char)
    print(f"Dataset contain {len(final_vocab_clean)} different characters")

    with open(output_vocab, 'w') as txt:
        for char in final_vocab_clean:
            txt.write(char + '\n')