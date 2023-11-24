import numpy as np
import sys
import time
import cv2
import re
import math

import tritonclient.grpc as grpcclient
from typing import Dict, List, Any, Union
from abc import ABC, abstractmethod
class BaseTriton(object):

    def __init__(self, input_name=[], input_type=[], input_dim=[], output_name=[], \
                url='', verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, static=False, **args):
        self.triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)

        self.input_name = input_name
        self.input_type = input_type
        self.input_dim = input_dim
        self.output_name = output_name
        self.client_timeout = client_timeout
        self.static = static

    def __call__(self, inputs_data, model_name: str):
        # Infer
        inputs = []
        outputs = []

        for i, input_data in enumerate(inputs_data):
            inputs.append(grpcclient.InferInput(self.input_name[i], input_data.shape, self.input_type[i]))
            inputs[i].set_data_from_numpy(input_data)
        for i, out_name in enumerate(self.output_name):
            outputs.append(grpcclient.InferRequestedOutput(out_name))

        # Test with outputs
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.client_timeout,
            headers={'test': '1'})
        
        if self.static:
            statistics = self.triton_client.get_inference_statistics(model_name=model_name)
            print(statistics)
            if len(statistics.model_stats) != 1:
                print("FAILED: Inference Statistics")
                sys.exit(1)

        # Get the output arrays from the results
        outputs_data = [results.as_numpy(self.output_name[i]) for i in range(len(self.output_name))]
        return outputs_data

class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        """
        Group the decoded characters and record the corresponding decoded positions. 

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters 
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words: 
                        - 'cn': continous chinese characters (e.g., 你好啊)
                        - 'en&num': continous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
                        The remaining characters in text are treated as separators between groups (e.g., space, '(', ')', etc.).
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection==True)[0]

        for c_i, char in enumerate(text):
            if '\u4e00' <= char <= '\u9fff':
                c_state = 'cn'
            elif bool(re.search('[a-zA-Z0-9]', char)):
                c_state = 'en&num'
            else:
                c_state = 'splitter'
            
            if char == '.' and state == 'en&num' and c_i + 1 < len(text) and bool(re.search('[0-9]', text[c_i+1])): # grouping floting number
                c_state = 'en&num'
            if char == '-' and state == "en&num": # grouping word with '-', such as 'state-of-the-art'
                c_state = 'en&num'
            
            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(self,
               text_index,
               text_prob=None,
               is_remove_duplicate=False,
               return_word_box=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection)
                result_list.append((text, np.mean(conf_list).tolist(), [
                    len(text_index[batch_idx]), word_list, word_col_list,
                    state_list
                ]))
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

class CPPDLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CPPDLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    continue
                if char_idx == '</s>':  # end
                    break
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
        
    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple):
            if isinstance(preds[-1], dict):
                preds = preds[-1]['align'][-1].numpy()
            else:
                preds = preds[-1].numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character

class CPPDModelBase(ABC):
    def __init__(self,  
                    ocr_vocab_path: str, 
                    input_shape=[3, 32, 768],
                    padding=True,
                    batch_size=8,
                    **kwargs) -> None:
        post_process_args = {"character_dict_path": ocr_vocab_path, "use_space_char": True}
        self.post_process = CPPDLabelDecode(**post_process_args)
        self.input_shape = input_shape
        self.padding = padding
        self.batch_size = batch_size

    @abstractmethod
    def forward(self, img_list: List[np.ndarray]) -> Any:
        pass

    @staticmethod
    def preprocess(img,
                    input_shape,
                    padding,
                    interpolation=cv2.INTER_LINEAR):
        imgC, imgH, imgW = input_shape
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=interpolation)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if input_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im
    
    def predict_batch(self, img_list):
        return self.predict(img_list)

    def predict(self, img_list: Union[np.ndarray, List[np.ndarray]]) -> List[Any]:
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]
        
        img_num = len(img_list)
        results = []
        for beg_img_no in range(0, img_num, self.batch_size):
            end_img_no = min(img_num, beg_img_no + self.batch_size)
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.preprocess(
                        img_list[ino], self.input_shape, 
                        self.padding)
                norm_img_batch.append(norm_img)
            norm_img_batch = np.asarray(norm_img_batch)
            output = self.forward(norm_img_batch)
            res = self.post_process(output)
            results.extend(res)
        return results

class CPPDModelTriton(CPPDModelBase):
    def __init__(self, ocr_vocab_path: str,
                 triton_params: Dict[str, Any],
                 input_shape: List[int] = [3, 32, 768],
                 padding: bool = True,
                 batch_size: int = 8,
                 **kwargs
                 ) -> None:
        super(CPPDModelTriton, self).__init__(ocr_vocab_path=ocr_vocab_path, 
                                            input_shape=input_shape,
                                            padding=padding,
                                            batch_size=batch_size)
        self.model = BaseTriton(**triton_params)
        self.model_name = triton_params['model_name']
    
    def forward(self, img_list: List[np.ndarray]):
        return self.model([img_list], model_name=self.model_name)[0]

class CPPDModelLegacy(CPPDModelBase):
    def __init__(self, ocr_vocab_path: str,
                 model_path: str,
                 input_shape: List[int] = [3, 32, 768],
                 padding: bool = True,
                 batch_size: int = 8,
                 use_gpu=False,
                 **kwargs
                 ) -> None:
        super(CPPDModelLegacy, self).__init__(ocr_vocab_path=ocr_vocab_path, 
                                            input_shape=input_shape,
                                            padding=padding,
                                            batch_size=batch_size)
        import onnxruntime as ort
        if use_gpu:
            self.model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        else:
            self.model = ort.InferenceSession(model_path)
    
    def forward(self, img_list: List[np.ndarray]):
        output = self.model.run(None, {self.model.get_inputs()[0].name: img_list})[0]
        return np.asarray(output)

if __name__ == '__main__':
    from pathlib import Path
    import time

    triton_params = {"input_name": ['x'], 
                     "input_type": ['FP32'], 
                     "output_name": ['softmax_15.tmp_0'], 
                     "model_name": 'rec',
                     "url": 'localhost:11004'
                     }
    ocr_vocab_path = "/mnt/ssd/martin/project/ocr/data/data_ocr_2/vocab.txt"
    legacy_model_path = "models/onnx/cppd_techainer.onnx"
    model = CPPDModelLegacy(ocr_vocab_path=ocr_vocab_path, 
                            model_path=legacy_model_path,
                            input_shape=[3, 32, 768], padding=True, 
                            use_gpu=True)

    test_folder = Path("/mnt/ssd/martin/project/ocr/data/debugs/merge")
    total_time = 0
    img_list = []
    for img in test_folder.iterdir():
        im = cv2.imread(str(img))
        img_list.append(im)
    for i in range(1):
        start = time.time()
        results = model.predict_batch(img_list)
        total_time += time.time() - start
    print(results)
    print("total time: ", total_time, " num images: ", len(list(test_folder.iterdir())))