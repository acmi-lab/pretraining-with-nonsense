import numpy as np
import os
from transformers import T5Tokenizer

class RandomVocab():
    def __init__(self, vocab_size = 5000):
        char_slots = np.ceil(np.log(vocab_size)/np.log(26))
        char_slots = int(char_slots)
        assert char_slots>0


        vocab = []
        for i in range(vocab_size):
            out=""
            val=i
            for j in range(char_slots):
                rem = val%26
                character = chr(97+rem)
                out = out + character
                val = int(val/26)

            vocab.append(out)

        assert len(set(vocab))==vocab_size
        self.tokens = vocab


class PretrainedT5Vocab():
    def __init__(self):
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        tokens = tokenizer.convert_ids_to_tokens(range(32000))
        tokens = tokens[3:]          # first 3 are ['<pad>', '</s>', '<unk>']
        self.tokens = tokens


