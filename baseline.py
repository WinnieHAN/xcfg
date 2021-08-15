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

SEEDS = [1213, 2324, 3435, 4546, 5657]
LANGS = ("ENGLISH", "CHINESE", "BASQUE", "GERMAN", "FRENCH", "HEBREW", "HUNGARIAN", "KOREAN", "POLISH", "SWEDISH") 

seed = SEEDS[0] 
random.seed(seed)
np.random.seed(seed)

def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn

class Node:
    def __init__(self, idx):
        self.idx = idx 
        self.span = [-1, -1]
        self.child = [None, None]

def build_spans(tree, l):
    if tree is None:
        return l, l 
    l, r = build_spans(tree.child[0], l)
    tree.span[0] = l
    l = r + 1
    l, r = build_spans(tree.child[1], l)
    tree.span[1] = r
    return tree.span[0], tree.span[1] 

def random_tree(n):
    nodes = [Node(0)]
    free_edges = [(0, 0), (0, 1)]
    for i in range(n - 1):
        father_idx, child_idx = random.choice(free_edges)
        assert nodes[father_idx].child[child_idx] == None

        node_idx = len(nodes)
        new_node = Node(node_idx)
        nodes[father_idx].child[child_idx] = new_node 
        nodes.append(new_node)

        free_edges.remove((father_idx, child_idx))
        free_edges.extend([(node_idx, 0), (node_idx, 1)])

    build_spans(nodes[0], 0)
    spans = [] 
    for node in nodes:
        l, r = -1, -1
        if node.child[0] is not None:
            l = node.child[0].idx 
        if node.child[1] is not None:
            r = node.child[1].idx 
        spans.append(node.span)
    spans.reverse()
    return spans

def lr_f1(per_label_f1, by_length_f1, corpus_f1, sent_f1, argmax_spans, spans, labels):
    pred = [(a[0], a[1]) for a in argmax_spans if a[0] != a[1]]
    pred_set = set(pred[:-1])
    gold = [(l, r) for l, r in spans if l != r] 
    gold_set = set(gold[:-1])

    tp, fp, fn = get_stats(pred_set, gold_set) 
    corpus_f1[0] += tp
    corpus_f1[1] += fp
    corpus_f1[2] += fn
    
    overlap = pred_set.intersection(gold_set)
    prec = float(len(overlap)) / (len(pred_set) + 1e-8)
    reca = float(len(overlap)) / (len(gold_set) + 1e-8)
    
    if len(gold_set) == 0:
        reca = 1. 
        if len(pred_set) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    sent_f1.append(f1)

    for j, gold_span in enumerate(gold[:-1]):
        label = labels[j]
        label = re.split("=|-", label)[0]
        per_label_f1.setdefault(label, [0., 0.]) 
        per_label_f1[label][0] += 1

        lspan = gold_span[1] - gold_span[0] + 1
        by_length_f1.setdefault(lspan, [0., 0.])
        by_length_f1[lspan][0] += 1

        if gold_span in pred_set:
            per_label_f1[label][1] += 1 
            by_length_f1[lspan][1] += 1

from batchify import get_nonbinary_spans

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '[':
            return True
        elif char == ']':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '[' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        # assert not(char == '[') and not(char == ']')
        output.append(char)
    return ''.join(output)



def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '[' or line_strip[i] == ']'
        if line_strip[i] == '[':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ']')
                i += 1
                while line_strip[i] != '[': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ']':
                    i += 1
                i += 1
                while line_strip[i] != ']' and line_strip[i] != '[':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ']' and line_strip[i] != '[':
                 i += 1
    assert i == max_idx
    return output_actions

def is_num_in_list(list_nums, NUM_SETS):
    for i in NUM_SETS[:len(list_nums)]:
        if i in list_nums:
            return True
    return False

def change_num_in_list(list_nums, NUM_SETS):
    s, e, leng = -1, -1, -1
    for i in NUM_SETS[:len(list_nums)]:
        leng = len(i)
        s = list_nums.find(i)
        if s>-1:
            NUM_SETS.pop(0)
            return s, s, s+leng-1
    return s, s, s+leng-1

def change_List(list_nums):
    NUM_SETS = [str(i) for i in range(300)]

    while (is_num_in_list(list_nums, NUM_SETS)):
        in_list, s, e = change_num_in_list(list_nums, NUM_SETS)
        if in_list:
            list_nums = list_nums[0:s] + '[T ' + list_nums[s:e+1] + ']' + list_nums[e+1:]

    # for i in range(len(list_nums)-1):
    #     if not list_nums[i] == '[' and not list_nums[i+1] == '[' and not list_nums[i+1] == 'N':
    #         list_nums = list_nums[0:i] + '[N ' + list_nums[i+1:]
    return list_nums

def tree_to_span(tree_index, lowercase=1, replace_num=1,  max_sent_l=0, shuffle=0, apply_length_filter=1):
    tree = ''.join(tree_index.__str__().split(','))
    tree = change_List(tree)
    # tree = '[NP [NN Woman] [VP [VBG gives][NN presentation]] [. .]]'
    #'[NP [NP [NN Woman]] [VP [VBG giving] [NP [DT a] [NN presentation]]] [. .]]'
        #'[S [S-TPC-1 [NP-SBJ [DT This]] [VP [ADVP [RB further]] [VBZ confuses] [NP [NNS retailers]]]] [NP-SBJ [PRP she]] [VP [VBZ says]]]'
    # '(S (S-TPC-1 (NP-SBJ (DT This)) (VP (ADVP (RB further)) (VBZ confuses) (NP (NNS retailers)))) (NP-SBJ (PRP she)) (VP (VBZ says)))'
    tree = tree.strip()
    action = get_actions(tree)
    span, binary_actions, nonbinary_actions = get_nonbinary_spans(action)
    return span

def lr_branching(ifile, btype=1):
    per_label_f1 = defaultdict(list) 
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    
    counter = Counter()
    with open(ifile, "r") as fr: 
        for line in fr:
            example_id = json.loads(line)['example_id']
            tree_word = json.loads(line)['tree']
            tree_index = json.loads(line)['tree_index_conll']
            caption = json.loads(line)['sentence']
            span = json.loads(line)['gold_spans']
            tag = words_to_tags(caption)
            label = tags_to_labels(tag)
            # (caption, span, label, tag) = json.loads(line)
            # caption = caption.strip().split()
            if len(caption) < 2:
                continue
            nword = len(caption)
            if btype == 0:
                token = "Left Branching: "
                argmax_span = [(0, r) for r in range(1, nword)] 
            elif btype == 1:
                token = "Right Branching: "
                argmax_span = [(l, nword - 1) for l in range(0, nword -1)] 
                argmax_span = argmax_span[1:] + argmax_span[:1]
            elif btype == 2:
                token = "Random Trees: "
                argmax_span = random_tree(nword - 1) 
                assert len(argmax_span) == nword - 1 
            elif btype == 3:
                token = "Predicted Trees: "
                argmax_span = tree_to_span(tree_index)
            lr_f1(per_label_f1, by_length_f1, corpus_f1, sent_f1, argmax_span, span, label)

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))

    token += ifile 
    print('\n{}\n\nCorpus F1: {:.4f}, Sentence F1: {:.4f}'.format(token, corpus_f1, sent_f1))

    f1_ids=["CF1", "SF1", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
    f1s = {"CF1": corpus_f1, "SF1": sent_f1}

    print("PER-LABEL-F1 (label, acc)\n")
    for k, v in per_label_f1.items():
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if True or k in f1_ids:
            f1s[k] = v[1] / v[0]
    # special case for SPMRL
    exist = len([x for x in f1_ids if x in f1s]) == len(f1_ids) 
    if not exist:
        xx = sorted(list(per_label_f1.items()), key=lambda x: -x[1][0])
        f1_ids = ["CF1", "SF1"] + [x[0] for x in xx[:8]]
    f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids] 
    print("\t".join(f1_ids))
    print(seed, " ".join(f1s))

    acc = []
    print("\nPER-LENGTH-F1 (length, acc)\n")
    xx = sorted(list(by_length_f1.items()), key=lambda x: x[0])
    for k, v in xx:
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if v[0] >= 5:
            acc.append((str(k), '{:.2f}'.format(v[1] / v[0])))
    k = [x for x, _ in acc]
    v = [x for _, x in acc]
    print(" ".join(k))
    print(" ".join(v))

def main_lr_branching_random_predicted(iroot):
    print("{} ENDS\n".format(iroot))
    for btype, name in enumerate(["LEFT-BRANCHING", "RIGHT-BRANCHING" "RANDOM", "PREDICTERD"]):
        # ifile = iroot + f"{lang.lower()}-train.json"
        ifile = iroot
        lr_branching(ifile, btype)
        print("\n")



def words_to_tags(words):

    # 分词
    text = ' '.join(words).lower()
    #"Sentiment analysis is a challenging subject in machine learning.\
     # People express their emotions in language that is often obscured by sarcasm,\
     #  ambiguity, and plays on words, all of which could be very misleading for \
     #  both humans and computers.".lower()
    text_list = nltk.word_tokenize(text)
    # rm punct
    # english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    # text_list = [word for word in text_list if word not in english_punctuations]
    # rm some words
    # stops = set(stopwords.words("english"))
    # text_list = [word for word in text_list if word not in stops]
    return [i[1] for i in nltk.pos_tag(text_list)]

def tags_to_labels(tags):
    labels_set = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
    # SBAR stands for Subordinate Clause (see here). In your case the subordinate clause starts with the subordinate conjunction After
    # Note:  these are the 'modified' tags used for Penn tree banking;
    # these are the tags used in the Jet system. NP, NPS, PP, and PP$ from the original Penn part-of-speech tagging were changed to NNP, NNPS, PRP, and PRP$ to avoid clashes with standard syntactic categories.

    tagslabel = {'CC': 'CC', 'CD': 'CD', 'DT': 'DT', 'EX': 'EX', 'FW': 'FW', 'IN': 'SBAR', 'JJ': 'ADJP', 'JJR': 'ADJP', 'JJS': 'ADJP', 'LS': 'LS', 'MD': 'MD', 'NN': 'NP', 'NNS': 'NP', \
                 'NNP': 'NP',  'NNPS': 'NP',  'PDT': 'PDT', 'POS': 'POS', 'PRP': 'PP', 'PRP$': 'PRP$', 'RB': 'ADVP', 'RBR': 'ADVP', 'RBS': 'ADVP', 'RP': 'RP', 'SYM': 'SYM', \
                 'TO': 'TO', 'UH': 'UH', 'VB': 'VP', 'VBD': 'VP', 'VBG': 'VP', 'VBN': 'VP', 'VBP': 'VP', 'VBZ': 'VP', 'WDT': 'WDT', 'WP': 'WP', 'WP$': 'WP$', 'WRB': 'WRB', \
                 'SBAR': 'SBAR'
                 }
    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '``',  "''"]
    return [tagslabel[i]  if i not in punctuations else 'PUNNCT' for i in tags]

def main_label_by_length(ifile, max_span_len=20, labels_set=["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]):
    """ Label distribution over constituent lengths.
        Return: (m x n) matrix: n labels and m lengths.
    """
    per_label_len = defaultdict(dict)
    lspans = list(range(2, max_span_len))
    with open(ifile, "r") as fr:
        for line in fr:
            example_id = json.loads(line)['example_id']
            tree_word = json.loads(line)['tree']
            tree_index = json.loads(line)['tree_index_conll']
            caption = json.loads(line)['sentence']
            spans = json.loads(line)['gold_spans']
            tag = words_to_tags(caption)
            labels = tags_to_labels(tag)
            # (caption, spans, labels, tag) = json.loads(line)

            spans = spans[:-1]
            labels = labels[:-1]
            for gold_span, label in zip(spans, labels):

                lspan = gold_span[1] - gold_span[0] + 1
                per_label_len.setdefault(lspan, defaultdict(float))

                # label = re.split("=|-", label)[0]
                per_label_len[lspan][label] += 1

    nspan = 0
    for k, v in per_label_len.items(): # k is length
        x = sum(v.values())
        nspan += x

    for k, v in per_label_len.items():
        for tag in v.keys():
            v[tag] = v[tag] / nspan

    # for k,v in per_label_len.items():
    #     print(k, v)
        # print(v)
        # print('\t'.join([str(item) for item in i]))

    data = []  # length*#labels
    # labels_set = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]
    for lspan in lspans:
        d = [per_label_len[lspan][label] for label in labels_set]
        data.append(d)

    data = np.array(data)
    print('\t'.join(labels_set))
    leng = 2
    for i in data:
        print(str(leng) + '\t' + '\t'.join([str(item) for item in i]))
        leng = leng + 1
    return data


if __name__ == '__main__':
    ifile = './data/parse_val_54.34.jsonl'
    #sys.argv[1]
    main_lr_branching_random_predicted(ifile)
    # main_label_by_length(ifile)

