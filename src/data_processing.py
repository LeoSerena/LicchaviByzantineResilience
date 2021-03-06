import re
import gc
import os
import logging
import pickle
import sys
import json
from typing import List, Union, Callable

from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import torch
import torchtext

sys.path.append('.')
from src.utils import make_dir_if_not_exists

def default_text_cleaner(string : str) -> str:
    """Cleans a given string.

    :param string: the string to be cleaned
    :type string: str
    :return: the cleaned string
    :rtype: str
    """    
    string = re.sub(r'-\n', '', string)
    string = re.sub(r"""[*#@&%£ö'ä$ü¨~^)('.+°¢=/><$\[\]`\-,:!?]""", '', string)
    string = re.sub(r'[0-9]', '', string)
    string = re.sub(' unk ', ' ', string)
    string = re.sub(' pad ', ' ', string)
    return string

def text_cleaner_raw(string) -> str:
    """Cleans a given string. The difference with
    the default is that it keeps dots <.> and removes
    sequences of more than one end of line or 2 or more
    spaces.

    :param string: the string to be cleaned
    :type string: str
    :return: the cleaned string
    :rtype: str
    """  
    string = re.sub(r'-\n', '', string)
    string = re.sub(r'\n+', ' ', string)
    string = re.sub(r"""[*#@&%£ö'ä$ü¨~^)('+°¢=/><$\[\]`\-,:!?`]""", '', string)
    string = re.sub(r'[0-9]', '', string)
    string = re.sub(' unk ', ' ', string)
    string = re.sub(' pad ', ' ', string)
    string = re.sub(r' {2,}', '', string)
    return string

class Vocabulary():
    def __init__(
        self,
        tokenizer : Callable[[str], List[str]],
        text_cleaner,
        max_voc_size : int = 10000,
        min_word_occ : int = 2
    ):
        """Vocabulary generated around textual data. Serves to
        convert ids to words and conversely.

        :param tokenizer: a tokenizer that, given raw textual data will
        generate a sequence of tokens
        :type tokenizer: str -> list[str]
        :param text_cleaner: cleaning function
        :type text_cleaner: str ->  str
        :param max_voc_size: maximum vocabulary size, defaults to 10000
        :type max_voc_size: int, optional
        :param min_word_occ: minimum number of times a wird most occur 
        to be taken into acount in the vocabulary, defaults to 2
        :type min_word_occ: int, optional
        :raises NotImplementedError: raises an exception if instantiated
        """    
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
        """builds the main access methods of the vocabulary, namely
        word_to_idx, vocab and idx_to_word.

        :param vocab: dictionary of words and their occurrences
        :type vocab: dict
        """
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

    def get_vocab_size(self) -> int:
        """Returns the length of the Vocabulary

        :rtype: int
        """
        return len(self.word_to_idx)

class FromRawTextVocabulary(Vocabulary):
    def __init__(
        self,
        text : str,
        **kwargs
    ):
        """Vocabulary builder from raw textual data

        :param text: input textual data
        :type text: str
        """
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
        """Vocabulary builder from list of texts

        :param tweets: list of textual inputs
        :type tweets: List[str]
        """    
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
        device : str,
        with_tqdm = True
    ):
        """Dataset class containing sequences of token ids each of same length.

        :param vocabulary: A vocabulary to map words to ids
        :type vocabulary: Vocabulary
        :param text: The text to create the dataset from
        :type text: Union[str, List[str]]
        :param min_seq_length: The minimum length for a sequence to be
        considered
        :type min_seq_length: int
        :param max_seq_length: The length of the sequences
        :type max_seq_length: int
        :param device: The device to store the sequence to
        :type device: str
        :param with_tqdm: Disaplays the sequence creation progress, defaults to True
        :type with_tqdm: bool, optional
        """
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
            for sentence in (tqdm(text) if with_tqdm else text)
        ]
        self.tokens = np.concatenate([
            self.pad_and_truncate(sequence) 
            for sequence in tokens 
            if len(sequence) > min_seq_length and sum(sequence) > 1
        ])
        
    def pad_and_truncate(self, sequence : List[int]) -> List[List[int]]:
        """Given a list of integers, will split it onto a list of lists of
        equal self.max_seq_length size and add 0s to the last one to fill it

        :param sequence: the list of token ids
        :type sequence: list[int]
        :return: a list of lists of token ids
        :rtype: list[list[int]]
        """
        sequence = np.array(sequence)
        # this makes the sequence the correct size by adding padding at the last one
        rest = len(sequence) % self.max_seq_length
        if rest > 1: # we verify that the last sequence is at least 2 words long
            sequence = np.concatenate((sequence, [self.vocabulary.padding_idx] * (self.max_seq_length - rest)))
        elif rest == 1: # otherwise we remove it
            sequence = sequence[:-1]
        return sequence.reshape(-1, self.max_seq_length)

    def token_len(self) -> int:
        """
        counts the number of tokens in the dataset
        """
        total = 0
        for tokens in self.tokens:
            total += sum(tokens > 0)
        return total

    def get_idx(self, token) -> int:
        """Gets the id of the token from the vocabulary

        :param token: a word token
        :type token: str
        :return: the corresponding id and 0 if not in the vocabulary
        :rtype: int
        """
        try:
            return self.vocabulary.word_to_idx[token]
        except KeyError:
            return self.vocabulary.padding_idx
        
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx]).to(self.device)
        
    def __len__(self):
        return len(self.tokens)



def prepare_tweets_data(
    N_USERS = 1000,
    data_path = 'data',
    id_ = 2,
    val_split = 0.2,
    test_split = 0.2,
    SEED = 23,
    nodes_data_folder = 'nodes_data'
):
    """Generates the data for pretraining the language model as
    well as the nodes data for federated learning.

    :param N_USERS: number of nodes data to generate, defaults to 1000
    :type N_USERS: int, optional
    :param data_path: where the input data is stored, defaults to 'data'
    :type data_path: str, optional
    :param id_: the file id to take as main, defaults to 2
    :type id_: int, optional
    :param val_split: split size of the validation set, defaults to 0.2
    :type val_split: float, optional
    :param test_split: split size of the test set, defaults to 0.2
    :type test_split: float, optional
    :param SEED: the numpy seed for shuffle, defaults to 23
    :type SEED: int, optional
    :param nodes_data_folder: where to save the nodes data, defaults to 'nodes_data'
    :type nodes_data_folder: str, optional
    """
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

        logging.info('generating nodes data...')
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
        logging.info('node data generated')

        # users_tweets.to_csv(os.path.join(data_path, user_tweets_file))
        logging.info('generated users data')
        del users_tweets
        gc.collect()

        np.random.seed(SEED)

        logging.info('generating language model datasets...')
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

def prepare_wiki_data(
    N_USERS : int,
    SEED : int,
    data_path : str,
    nodes_data_folder : str,
    data_name : str
):
    """prepares the pretrain and nodes data for the WikiText103 dataset.

    :param N_USERS: [description]
    :type N_USERS: [type]
    :param SEED: [description]
    :type SEED: [type]
    :param data_path: [description]
    :type data_path: [type]
    :param nodes_data_folder: [description]
    :type nodes_data_folder: [type]
    :param data_name: [description]
    :type data_name: [type]
    """
    np.random.seed(SEED)
    # Whether to use WikiText-2 or WikiText103
    # Path to data folder

    if '3' in data_name:
        name = 'wikitext-3'
        id_ = '103'
        path = os.path.join('.', data_path, name)
        make_dir_if_not_exists(path)
        train, val, test = torchtext.datasets.WikiText103(root = path, split = ('train', 'valid', 'test'))
    else:
        name = 'wikitext-2'
        id_ = '2'
        path = os.path.join('.', data_path, name)
        make_dir_if_not_exists(path)
        train, val, test = torchtext.datasets.WikiText2(root = path, split = ('train', 'valid', 'test'))
    
    train_set_file = f'train_{id_}.pickle'
    val_set_file = f'val_{id_}.pickle'
    test_set_file = f'test_{id_}.pickle'

    # splits train-val-test by article
    heading_pattern = r'( \n\n = [^=]*[^=] = \n\n )'

    raw_train = '\n'.join([x for x in train])
    train_articles = [x for x in re.split(heading_pattern, raw_train) if (len(x) > 200 and '\n' in x)]
    np.random.shuffle(train_articles)
    nodes_data = train_articles[:N_USERS]
    train_articles = train_articles[N_USERS:]

    raw_val = '\n'.join([x for x in val])
    val_articles = [x for x in re.split(heading_pattern, raw_val) if len(x) > 200]
    raw_test = '\n'.join([x for x in test])
    test_articles = [x for x in re.split(heading_pattern, raw_test) if len(x) > 200]

    # 1 node = 1 article
    k = 0
    for (i,art) in enumerate(nodes_data):
        art = [x for x in re.split(r'\n', art) if len(x) > 20]
        while len(art) < 100:
            art = train_articles[k]
            art = [x for x in re.split(r'\n', art) if len(x) > 200]
            train_articles = train_articles[1:]
            k+=1
        with open(os.path.join(nodes_data_folder, f"node_{i+1}_{len(art)}.pickle"), 'wb') as f:
            pickle.dump(art, f)

    # store data as raw text
    with open(os.path.join(path, train_set_file), 'wb') as f:
        pickle.dump(train_articles, f)
    with open(os.path.join(path, val_set_file), 'wb') as f:
        pickle.dump(val_articles, f)
    with open(os.path.join(path, test_set_file), 'wb') as f:
        pickle.dump(test_articles, f)
    
    logging.info(f'generated train-val-test sets for id {id_}')
    logging.info(f"""
        trainining set : {len(train_articles)} articles
        validation set : {len(val_articles)} articles
        test set       : {len(test_articles)} articles
    """)

if __name__ == '__main__':
    logging.basicConfig(filename='logs/logs.log', level=logging.DEBUG)
    if sys.argv[1] == 'wiki':
        with open('./config_files/CONFIG_FEDERATED_WIKI.json', 'r') as f:
            federated_parameters = json.load(f)
        with open('./config_files/CONFIG_MODEL_WIKI.json', 'r') as f:
            model_parameters = json.load(f)
    if sys.argv[1] == 'tweet':
        with open('./config_files/CONFIG_FEDERATED_TWEETS.json', 'r') as f:
            federated_parameters = json.load(f)
        with open('./config_files/CONFIG_MODEL_TWEETS.json', 'r') as f:
            model_parameters = json.load(f)
    
    make_dir_if_not_exists('nodes_data')
    nodes_folder = os.path.join('nodes_data', federated_parameters['nodes_data_folder'])
    make_dir_if_not_exists(nodes_folder)
    data_parameters = model_parameters['DATA_PARAMETERS']
    if data_parameters['data_name'] == 'tweets':
        id_ = sys.argv[2]
        prepare_tweets_data(
            N_USERS = federated_parameters['num_nodes'],
            data_path = data_parameters['data_folder'],
            id_ = id_,
            val_split = data_parameters['val_split'],
            test_split = data_parameters['test_split'],
            SEED = model_parameters['NUMPY_SEED'],
            nodes_data_folder = nodes_folder
        )
    elif 'WikiText' in data_parameters['data_name']:
        prepare_wiki_data(
            N_USERS = federated_parameters['num_nodes'],
            data_path = data_parameters['data_folder'],
            SEED = model_parameters['NUMPY_SEED'],
            nodes_data_folder = nodes_folder,
            data_name = data_parameters['data_name']
        )