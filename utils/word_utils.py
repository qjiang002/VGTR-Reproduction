# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re
import torch
import codecs
import pandas as pd
import numpy as np

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
END_TOKEN = '<eos>'
CNT_SPECIAL_TOKEN = 3

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')



class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx




class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def set_max_len(self, value):
        self.max_len = value


    def build_dict(self, sentences):  # A list of strings, each being a sentence
        for sentence in sentences:
            sentence = sentence.strip()
            self.add_to_corpus(sentence)
        self.dictionary.add_word(UNK_TOKEN)
        self.dictionary.add_word(PAD_TOKEN)
        self.dictionary.add_word(END_TOKEN)

    def build_embedding(self, embedding_file: str, dictionary: Dictionary, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.num_embeddings = len(dictionary)

        # If no embedding given
        if embedding_file is None:
            self.embedding = None
            return 

        glove = pd.read_csv(embedding_file, sep=" ", quoting=3, header=None, index_col=0)
        glove_embedding = {key: val.values for key, val in glove.T.items()}  # {word: vec} dictionary
        vocab = dictionary.idx2word  # a list of words

        embedding = np.zeros((len(vocab) + CNT_SPECIAL_TOKEN, embedding_dim))

        for word_idx, word in enumerate(dictionary.idx2word):
            if word in glove_embedding:
                embedding[word_idx] = glove_embedding[word]

        self.embedding = embedding


    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=20):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())
        # words = [w.lower() for w in words if len(w) > 0]
        words = [w.lower() for w in words if (len(w) > 0 and w!=' ')]   ## do not include space as a token

        if words[-1] == '.':
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                # words = [PAD_TOKEN] * (max_len - len(words)) + words
                words = words + [END_TOKEN] + [PAD_TOKEN] * (max_len - len(words) - 1)

        tokens = len(words) ## for end token
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            # print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii','ignore').decode('ascii')))
            if type(word)!=type('a'):
                print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii','ignore').decode('ascii')))
                word = word.encode('ascii','ignore').decode('ascii')
            ids[token] = self.dictionary[word]
            token += 1
        # ids[token] = self.dictionary[END_TOKEN]
        return ids

    def __len__(self):
        return len(self.dictionary)



    def get_embedding_matrix(self):
        return self.embedding

