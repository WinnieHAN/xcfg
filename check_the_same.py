import os, sys, re
import pickle, json
import numpy as np
import random

import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
import numpy as np


from nltk import Tree
from tqdm import tqdm
from collections import Counter, defaultdict

def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)

class JSONLReader(object):
    def __init__(self, lowercase=True, filter_length=0, delim=' ', include_id=False):
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0

    def read(self, filename):
        sentences = []

        # extra
        extra = dict()
        example_ids = []
        trees = []

        # read
        with open(filename) as f:
            for line in tqdm(f, desc='read'):
                ex = json.loads(line)
                s = ex[0]
                sentences.append(s)

        extra['example_ids'] = example_ids
        extra['trees'] = trees

        return sentences

def words_to_tags(words):
    text = ' '.join(words).lower()
    text_list = nltk.word_tokenize(text)
    return [i[1] for i in nltk.pos_tag(text_list)]

def main_label_by_length_new(ifile, tfile):
    """ Label distribution over constituent lengths.
        Return: (n x m) matrix: n labels and m lengths.
    """

    stc_set = []
    with open(ifile, "r") as fr:
        for line in fr:
            caption = ' '.join(json.loads(line)['sentence'])
            stc_set.append(caption)
    stc_set = set(stc_set)


    stc_set1 = []
    with open(tfile, "r") as fr1:
        for line in fr1:
            caption1 = ' '.join(json.loads(line)['sentence'])
            stc_set1.append(caption1)
    data = stc_set1
    # json_reader = JSONLReader()
    # data = json_reader.read(tfile)

    for i in range(len(data)):
        if not data[i] in stc_set:
            print(i)
            print(data[i])


if __name__ == '__main__':
    tfile = '/home/wenjuan/Code/xcfg/data/parse.jsonl'                 # flickr_val.json  E:/GitHub/xcfg/data/parse.jsonl'

    ifile = '/home/wenjuan/Code/xcfg/data/parse_59p4.jsonl'
    #sys.argv[1]
    # main_lr_branching(ifile)
    # main_label_by_length_new(ifile, tfile)
    main_label_by_length_new(ifile, tfile)
