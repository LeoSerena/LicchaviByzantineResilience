import re

import nltk

def text_cleaning(string):
    string = re.sub('-\n', '', string)
    string = re.sub(r"""[*#@&%£ö'ä$ü¨~^)('+°¢./><$\[\]`,:!?]""", '', string)
    string = re.sub('[0-9]', '', string)
    return re.sub('\n', ' ', string)

def prepare_sentences(text, max_sentence_length):
    sentences = nltk.sent_tokenize(text)
    sentences = [re.sub(r".*: ", '', sent, 1) for sent in sentences]
    sentences = [text_cleaning(sentence) for sentence in sentences]
    sentences = [[w.lower() for w in nltk.word_tokenize(sentence)] for sentence in sentences]
    return [sent for sent in sentences if len(sent) < max_sentence_length]