from abc import abstractclassmethod
import re
import gc
import os
import logging
import pickle
import sys
import json
from typing import List, Union

from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import torch

def default_text_cleaner(string):
    string = re.sub(r'-\n', '', string)
    string = re.sub(r"""[*#@&%£ö'ä$ü¨~^)('.+°¢=/><$\[\]`\-,:!?]""", '', string)
    string = re.sub(r'[0-9]', '', string)
    string = re.sub('unk', ' ', string)
    string = re.sub('pad', ' ', string)
    return string

class Vocabulary():
    def __init__(
        self,
        tokenizer,
        text_cleaner,
        max_voc_size : int = 20000,
        min_word_occ : int = 2
    ):
        if self.__class__ == Vocabulary:
            raise NotImplementedError(
        """
        This is an abstract class.
        To instanciate a Vocabulary, use the FromRawTextVocabulary 
        or the FromTweetsVocabulary class
        """)
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = nltk.sent_tokenize
        if text_cleaner is not None:
            self.text_cleaner = text_cleaner
        else:
            self.text_cleaner = default_text_cleaner
        
        self.padding_token, self.padding_idx = 'pad', 0
        self.unknown_token, self.unknown_idx = 'unk', 1
        self.min_word_occ = min_word_occ
        self.max_voc_size = max_voc_size

    def build_vocab(self, vocab):
        # sort voc and remove words not occuring enough
        self.vocab = {
            k: v 
            for (k, v) in sorted(vocab.items(), key=lambda item: -item[1])
            if v >= self.min_word_occ
        }
        logging.info('vocabulary sorted')
        # keep only top words
        self.vocab = {
            k : v for i, (k,v) in enumerate(self.vocab.items()) if i < (self.max_voc_size - 2)
        }
        
        self.word_to_idx = {k : (i+2) for i,(k,_) in enumerate(self.vocab.items())}
        self.word_to_idx[self.padding_token] = self.padding_idx
        self.vocab[self.padding_token] = 1
        self.word_to_idx[self.unknown_token] = self.unknown_idx
        self.vocab[self.unknown_token] = 1
        self.idx_to_word = {v : k for k, v in self.word_to_idx.items()}

        logging.info('vocabulary built')

    def get_vocab_size(self):
        return len(self.word_to_idx)

class FromRawTextVocabulary(Vocabulary):
    def __init__(
        self,
        text : str,
        **kwargs
    ):
        super(FromRawTextVocabulary, self).__init__(**kwargs)
        text = self.text_cleaner(text)
        tokens = self.tokenizer(text)
        vocab = {}
        for token in tokens:
            if token in vocab.keys():
                vocab[token] += 1
            else:
                vocab[token] = 1
        self.build_vocab(vocab)

class FromTweetsVocabulary(Vocabulary):
    def __init__(
        self,
        tweets : List[str],
        **kwargs
    ):
        super(FromTweetsVocabulary, self).__init__(**kwargs)
        vocab = {}
        for tweet in tqdm(tweets):
            for token in self.tokenizer(tweet):
                if token.isalpha():
                    if token in vocab.keys():
                        vocab[token] += 1
                    else:
                        vocab[token] = 1
        self.build_vocab(vocab)
        

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vocabulary : Vocabulary,
        text : Union[str, List[str]],
        min_seq_length : int,
        max_seq_length : int,
        device : str
    ):
        self.vocabulary = vocabulary
        self.max_seq_length = max_seq_length
        self.device = device
        if isinstance(text, str):
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r' {2,}', ' ', text)
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
        self.tokens = np.concatenate([
            self.pad_and_truncate(sequence) 
            for sequence in tokens 
            if len(sequence) > min_seq_length and sum(sequence) > 1
        ])
        
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
            return self.vocabulary.padding_idx
        
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx]).to(self.device)
        
    def __len__(self):
        return len(self.tokens)

def prepare_data(
    N_USERS = 1000,
    data_path = 'data',
    id_ = 2,
    val_split = 0.2,
    test_split = 0.2,
    SEED = 23,
    nodes_data_folder = 'nodes_data'
):
    tweets_file = f'tweets_{id_}.csv'

    # user_tweets_file = f'users_tweets_{id_}.csv'

    train_set_file = f'train_{id_}.pickle'
    val_set_file = f'val_{id_}.pickle'
    test_set_file = f'test_{id_}.pickle'
    
    data_files = os.listdir(data_path)

    if tweets_file not in data_files:
        logging.error(f'Neither parsed nor raw data was found in the {data_path} directory')
        print(tweets_file)
        print(data_files)
    else:
        users_tweets = pd.DataFrame()
        tweets = pd.read_csv(os.path.join(data_path, tweets_file))
        tweets = tweets[tweets['lang'] == 'en']
        lengths = tweets['body'].map(len)
        tweets = tweets[lengths.map(lambda x : 20 < x and x <= 140)]

        users_ids = list(tweets['author_id'].value_counts()[:N_USERS].index)
        users_tweets = users_tweets.append(tweets[tweets['author_id'].map(lambda x : x in users_ids)])
        for j in range(1,4):
            if j != id_:
                tweets = pd.read_csv(os.path.join(data_path,  f'tweets_{j}.csv'))
                tweets = tweets[tweets['lang'] == 'en']
                lengths = tweets['body'].map(len)
                tweets = tweets[lengths.map(lambda x : 20 < x and x <= 140)]
                users_tweets = users_tweets.append(tweets[tweets['author_id'].map(lambda x : x in users_ids)])

        for (user_id, screename), user_tweets in users_tweets.groupby(['author_id', 'author_screen_name']):
            i = int(users_ids.index(user_id) + 1)
            bodies = list(user_tweets['body'])
            with open(os.path.join(nodes_data_folder, f"node_{i}_{len(bodies)}_{user_id}_{screename}.pickle"), 'wb') as f:
                pickle.dump(bodies, f)

        # users_tweets.to_csv(os.path.join(data_path, user_tweets_file))
        logging.info('generated users data')
        del users_tweets
        gc.collect()

        np.random.seed(SEED)

        tweets = pd.read_csv(os.path.join(data_path, tweets_file))
        tweets = tweets[tweets['lang'] == 'en']
        lengths = tweets['body'].map(len)
        tweets = tweets[lengths.map(lambda x : 20 < x and x <= 140)]
        LM_tweets = tweets[tweets['author_id'].map(lambda x : x not in users_ids)].set_index('author_id')
        del tweets
        gc.collect()
        LM_tweets_users = list(LM_tweets.index.unique())
        np.random.shuffle(LM_tweets_users)

        n_users = len(LM_tweets_users)
        test_users = LM_tweets_users[:int(test_split * n_users)]
        val_users = LM_tweets_users[int(test_split * n_users):int((test_split + val_split) * n_users)]
        train_users = LM_tweets_users[int((test_split + val_split) * n_users):]

        assert n_users == len(test_users) + len(val_users) + len(train_users)

        test_set = list(LM_tweets.loc[test_users]['body'])
        val_set = list(LM_tweets.loc[val_users]['body'])
        train_set = list(LM_tweets.loc[train_users]['body'])

        assert len(LM_tweets) == len(test_set) + len(val_set) + len(train_set)

        with open(os.path.join(data_path, train_set_file), 'wb') as f:
            pickle.dump(train_set, f)
        with open(os.path.join(data_path, val_set_file), 'wb') as f:
            pickle.dump(val_set, f)
        with open(os.path.join(data_path, test_set_file), 'wb') as f:
            pickle.dump(test_set, f)
        
        logging.info(f'generated train-val-test sets for id {id_}')
        logging.info(f"""
            trainining set : {len(train_set)} tweets for {len(train_users)} users
            validation set : {len(val_set)} tweets for {len(val_users)} users
            test set       : {len(test_set)} tweets for {len(test_users)} users
        """)

if __name__ == '__main__':

    logging.basicConfig(filename='logs/logs.log', level=logging.DEBUG)

    if len(sys.argv) != 2:
        raise AssertionError("""required 1 argument: the data id""")
    id_ = sys.argv[1]

    with open('CONFIG_FEDERATED.json', 'r') as f:
        federated_parameters = json.load(f)
    with open('CONFIG_MODEL.json', 'r') as f:
        model_parameters = json.load(f)

    data_parameters = model_parameters['DATA_PARAMETERS']

    prepare_data(
        N_USERS = federated_parameters['num_nodes'],
        data_path = data_parameters['data_folder'],
        id_ = id_,
        val_split = data_parameters['val_split'],
        test_split = data_parameters['test_split'],
        SEED = model_parameters['NUMPY_SEED'],
        nodes_data_folder = federated_parameters['nodes_data_folder']
    )