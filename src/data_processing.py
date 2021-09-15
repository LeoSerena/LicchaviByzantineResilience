import re
from typing import List, Union

import numpy as np
import nltk
import torch

def default_text_cleaner(string):
    string = re.sub('-\n', '', string)
    string = re.sub(r"""[*#@&%£ö'ä$ü¨~^)('.+°¢=/><$\[\]`\-,:!?]""", '', string)
    string = re.sub('[0-9]', '', string)
    return re.sub('\n', ' ', string)

class Vocabulary():
    def __init__(
        self,
        tokenizer,
        text_cleaner,
        max_voc_size : int = 20000,
        min_word_occ : int = 2
    ):
        if self.__class__ == 'Vocabulary':
            raise NotImplementedError("""This is an abstract class""")
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = nltk.sent_tokenize
        if text_cleaner is not None:
            self.text_cleaner = text_cleaner
        else:
            self.text_cleaner = default_text_cleaner
        
        self.padding_token, self.padding_idx = '<PAD>', 0
        self.unknown_token, self.unknown_idx = '<UNK>', 0
        self.min_word_occ = min_word_occ
        self.max_voc_size = max_voc_size

    def build_vocab(self, text):
        text = self.text_cleaner(text)
        vocab = {}
        for token in self.tokenizer(text):
            if token in vocab.keys():
                vocab[token] += 1
            else:
                vocab[token] = 1

        # sort voc and remove words not occuring enough
        self.vocab = {
            k: v 
            for (k, v) in sorted(vocab.items(), key=lambda item: -item[1])
            if v >= self.min_word_occ
        }
        # keep only top words
        self.vocab = {
            k : v for i, (k,v) in enumerate(self.vocab.items()) if i < (self.max_voc_size - 2)
        }
        
        self.word_to_idx = {k : (i+1) for i,(k,_) in enumerate(self.vocab.items())}
        self.word_to_idx[self.padding_token] = self.padding_idx
        self.vocab[self.padding_token] = 1
        self.word_to_idx[self.unknown_token] = self.unknown_idx
        self.vocab[self.unknown_token] = 1
        self.idx_to_word = {v : k for k, v in self.word_to_idx.items()}

    def get_vocab_size(self):
        return len(self.word_to_idx)

class FromRawTextVocabulary(Vocabulary):
    def __init__(
        self,
        text : str,
        **kwargs
    ):
        super(FromRawTextVocabulary, self).__init__(**kwargs)
        self.build_vocab(text)

class FromTweetsVocabulary(Vocabulary):
    def __init__(
        self,
        tweets : List[str],
        **kwargs
    ):
        super(FromTweetsVocabulary, self).__init__(**kwargs)
        self.build_vocab(tweets)

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vocabulary : Vocabulary,
        text : Union[str, List[str]],
        max_seq_length : int,
        device : str
    ):
        self.vocabulary = vocabulary
        self.max_seq_length = max_seq_length
        self.device = device
        if isinstance(text, str):
            text = nltk.sent_tokenize(text)
        elif isinstance(text, List):
            if not isinstance(text[0], str):
                print("text should be either a list of str or a str")
        else:
            print("text should be either a list of str or a str") 
        
        tokens = [
            [self.get_idx(w) for w in vocabulary.tokenizer(vocabulary.text_cleaner(sentence))]
            for sentence in text
        ]
        self.tokens = np.concatenate([self.pad_and_truncate(sequence) for sequence in tokens if len(sequence) > 1])
        
    def pad_and_truncate(self, sequence):
        sequence = np.array(sequence)
        # this makes the sequence the correct size by adding padding at the last one
        rest = len(sequence) % self.max_seq_length
        if rest > 1: # we verify that the last sequence is at least 2 words long
            sequence = np.concatenate((sequence, [self.vocabulary.padding_idx] * (self.max_seq_length - rest)))
        elif rest == 1: # otherwise we remove it
            sequence = sequence[:-1]
        return sequence.reshape(-1, self.max_seq_length)

    
    def get_idx(self, token):
        try:
            return self.vocabulary.word_to_idx[token]
        except KeyError:
            return self.vocabulary.unknown_idx
        
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx]).to(self.device)
        
    def __len__(self):
        return len(self.tokens)