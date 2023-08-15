# coding=utf-8
import os
import re
import codecs
from copy import deepcopy
from itertools import groupby
from pathlib import PurePath

from torch.utils.data import Dataset

import second.args as args

NMT_SEP_ID = 0
NMT_PAD_ID = 1
has_simple = True

def sorted_nt(data_list, key=str.upper, reverse=False):
    keys = {
        '!': 0, '#': 1, '$': 2, '%': 3, '&': 4, '(': 5, ')': 6, '+': 7, ',': 8, '-': 9, '.': 10, '0': 11, '1': 12, '2': 13, '3': 14, '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20, ';': 21, '=': 22, '@': 23, 'A': 24, 'B': 25, 'C': 26, 'D': 27, 'E': 28, 'F': 29, 'G': 30, 'H': 31, 'I': 32, 'J': 33, 'K': 34, 'L': 35, 'M': 36, 'N': 37, 'O': 38, 'P': 39, 'Q': 40, 'R': 41, 'S': 42, 'T': 43, 'U': 44, 'V': 45, 'W': 46, 'X': 47, 'Y': 48, 'Z': 49, '[': 50, ']': 51, '^': 52, '_': 53, '`': 54, '{': 55, '}': 56, '~': 57
    }
    return sorted(data_list, key=lambda x: [keys.get(c, ord(c)) for c in key(x)], reverse=reverse)

def my_collate(data_list):
    sents = []
    cases = []
    types = []
    for data in data_list:
        sents.append(data['token'])
        cases.append(data['case'])
        types.append(data['label'])
    return {'token': sents, 'case': cases, 'label': types}

class MyDataset(Dataset):
    def __init__(self, tokens, cases, labels):
        self.tokens = tokens
        self.cases = cases
        self.labels = labels
    
    def __getitem__(self, index):
        tokens = self.tokens[index]
        cases = self.cases[index]
        labels = self.labels[index]
        return {'token': tokens, 'case': cases, 'label': labels}

    def __len__(self):
        return len(self.tokens)


def proc_func(label):
    newstr = deepcopy(label)

    idx = -1
    idx_list = []
    while(idx < len(label)):
        idx = label.find('->', idx+1)
        if idx == -1:
            break
        assert label[idx-1] == ')' and label[idx+2] == '(', label

        left = -1
        right = -1
        lp = 0
        rp = 1
        for i in range(idx-2, -1, -1):
            if label[i] == '(':
                lp += 1
            elif label[i] == ')':
                rp += 1
            if lp == rp:
                left = i
                break
        
        lp = 1
        rp = 0
        for i in range(idx+3, len(label)):
            if label[i] == '(':
                lp += 1
            elif label[i] == ')':
                rp += 1
            if lp == rp:
                right = i
                break

        idx_list.append((left, idx, right))
    
    idx_list.sort(key=lambda x: x[0])
    while idx_list:
        left, idx, right = idx_list.pop(0)
        param = label[left:idx]
        ret = label[idx+3:right]
        orig = label[left:right+1]
        new = '-> ' + ret + param
        newstr = newstr.replace(orig, new, 1)
    
    return newstr

def proc_struct(label):
    if label.find('struct') >= 0:
        label = re.sub(r'struct\)', 'struct())', label)
        label = re.sub(r'struct\,', 'struct(),', label)
    if label.find('union') >= 0:
        label = re.sub(r'union\)', 'union())', label)
        label = re.sub(r'union\,', 'union(),', label)
    
    return label

def proc_long(label):
    if label.find('long long') >= 0:
        label = label.replace('long long', 'longlong')
    if label.find('long double') >= 0:
        label = label.replace('long double', 'longdouble')
    
    return label

def proc_enum(label):
    if label.find('enum,') >= 0:
        label = re.sub(r'enum\,', 'enum(),', label)
        label = re.sub(r'enum\)', 'enum())', label)
    
    return label

def proc_label(label):
    if('(' not in label):
        return [label]
    
    # process func struct enum long long
    label = proc_enum(label)
    label = proc_func(label)
    label = proc_struct(label)
    label = proc_long(label)

    # omit recurrence
    label = label.replace('`', ' <eos>')

    # replace all delimiter with space
    label = label.replace('(', ' ')
    label = label.replace(',', ' ')
    label = label.replace(')', ' <eos> ')

    # replace space
    label = re.sub(r'\s+', ' ', label)
    label = re.sub(r'^\s', '', label)
    label = re.sub(r'\s$', '', label)

    # recover long long
    # label = re.sub(r'long\tlong', 'long long', label)
    # label = re.sub(r'long\tdouble', 'long double', label)

    # e.g. 
    # input: (int, *(char))->(long)
    # output: [int, *, char, ->, long]
    return label.split(' ')

def proc_casing(casing):
    if casing == 'pointer':
        casing = '*'
    elif casing == 'function':
        casing = '->'
    elif casing == 'long long':
        casing = 'longlong'
    elif casing == 'long double':
        casing = 'longdouble'
    return casing

def read_data_file(path):
    data_list = []
    for item in sorted_nt(os.listdir(path)):
        name = str(PurePath(path) / item)
        if(item[0] != '.' and os.path.isfile(name)): 
            with open(name, 'rb') as file_r:
                data_f = file_r.read().decode('utf-8').strip()
            data_list.append(data_f)
    
    return data_list

def read_vocab_file(path):
    vocab = {}
    with codecs.open(path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode('utf-8').strip()
            #if (line.endswith(u'\n')):
            #line = line[:-1]
            vocab[line] = len(vocab)
    return vocab

def read_unseen_file(path):
    unseen = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            unseen.append(int(line))
    return unseen

def get_data(sim_path, cmp_path, max_len, mode="test"): 
    labels = []
    cases = []
    seq = []
    counts = []
    taglens = []
    sim_list = read_data_file(sim_path)
    cmp_list = read_data_file(cmp_path)

    if(mode == "train"):
        seq_num = []
        max_count = []
        batch_length = [50, 100, 200, 400, 600, max_len]
        batch_size = [128, 64, 32, 16, 8, 4]
        for i in range(len(batch_length)):
            labels.append([])
            cases.append([])
            seq.append([])
            counts.append([])
    
    for item1, item2 in zip(sim_list, cmp_list):
        # each file
        item1 = item1.strip().split('\n')
        item2 = item2.strip().split('\n')
        assert len(item1) == len(item2)
        if(len(item1) > max_len):
            continue
        words = []
        casing = []
        label = []
        count = 0
        lens = []
        for data1, data2 in zip(item1, item2):
            # each line
            data1 = data1.strip().rsplit('\t', 1)
            data2 = data2.strip().rsplit('\t', 1)
            if(len(data1) <= 1):
                continue
                
            assert data1[0] == data2[0]
            count += 1
            words.append(data1[0])
            casing.append(proc_casing(data1[1]))
            label_list = proc_label(data2[1])
            label.append(label_list)
            lens.append(len(label_list))
            assert len(words) == len(casing) == len(label)
            if(len(words) == max_len):
                break
        
        if(len(words) > 0):
            if(mode == "train"):
                i = [idx for idx in range(len(batch_length)) if batch_length[idx] >= len(words)][0]
                seq[i].append(words)
                cases[i].append(casing)
                labels[i].append(label)
                counts[i].append(count)
            else:
                seq.append(words)
                cases.append(casing)
                labels.append(label)
                counts.append(count)
                taglens.append(lens)
    
    if(mode == "train"):
        for i in range(len(batch_length)):
            seq_num.append(len(seq[i]))
            max_count.append(max(counts[i]))
    
        # seq           根据句子长度区分的所有token序列
        # cases         根据句子长度区分的所有简单类型序列
        # labels        根据句子长度区分的所有标签序列
        # seq_num       每个batch区间内部句子的个数
        # max_count     每个batch区间内句子长度最大值
        # batch_length  每个batch区间句子的最大长度
        # batch_size    每次训练batch个数
        return seq, cases, labels, counts, max_count, batch_length, batch_size
    else:
        # seq           所有token序列
        # cases         所有简单类型序列
        # labels        所有标签序列
        # max(counts)   句子长度
        return seq, cases, labels, counts, taglens

def gen_vocab(src_vocab_file, tgt_vocab_file):
    src_vocab = read_vocab_file(src_vocab_file)
    tgt_vocab = read_vocab_file(tgt_vocab_file)
    tgt_vocab_rev = {v: k for k, v in tgt_vocab.items()}

    return src_vocab, tgt_vocab, tgt_vocab_rev

def process_train_data(seqs_list, cases_list, tags_list, max_lens, max_tag, src_vocab, tgt_vocab):
    assert len(seqs_list) == len(cases_list) == len(tags_list) == len(max_lens)
    for i, (seqs, cases, tags, max_len) in enumerate(zip(seqs_list, cases_list, tags_list, max_lens)):
        # 每个长度区间
        seqs, cases, tags = process_data(seqs, cases, tags, max_len, max_tag, src_vocab, tgt_vocab)
        # seqs_list[i] = seqs
        # tags_list[i] = tags
    
    return seqs_list, cases_list, tags_list

def process_data(seqs, cases, tags, max_len, max_tag, src_vocab, tgt_vocab):
    assert len(seqs) == len(cases) == len(tags)
    for i, (seq, case, tag) in enumerate(zip(seqs, cases, tags)):
        # 每个文件
        assert len(seq) == len(case) == len(tag)
        for idx, (token, casing, label) in enumerate(zip(seq, case, tag)):
            # 文件内每行
            seq[idx] = src_vocab[token] if token in src_vocab else 0
            if has_simple:
                case[idx] = tgt_vocab[casing] if casing in tgt_vocab else 0
            else:
                # case[idx] = 5
                case[idx] = tgt_vocab['O'] if 'O' in tgt_vocab else 0
            tag[idx] = [tgt_vocab[s] if s in tgt_vocab else 0 for s in label]
            if len(tag[idx]) < max_tag:
                tag[idx] += [0] * (max_tag-len(tag[idx]))
            else:
                tag[idx] = tag[idx][:max_tag]
        assert len(seq) == len(case) == len(tag)
        if(len(seq) < max_len):
            seq += [NMT_PAD_ID] * (max_len-len(seq))
            case += [NMT_PAD_ID] * (max_len-len(case))
            tag += [[0] * max_tag] * (max_len-len(tag))
    
    # still need torch embedding
    return seqs, cases, tags

def process_result(preds, seq_lens, tag_lens, tgt_vocab_rev):
    # each batch
    out_list = []
    for i, pred in enumerate(preds):
        # each file
        pred_str_list = []
        for j, tags in enumerate(pred):
            # each pred tag list
            if j >= seq_lens[i]:
                break
            pred_list = []
            for tag in tags:
                pred_list.append(tgt_vocab_rev[tag])
            pred_list = pred_list[:tag_lens[i][j]]
            pred_str_list.append('\t'.join(pred_list))
        
        # pred_str_list = pred_str_list[:seq_lens[i]]
        assert len(pred_str_list) == seq_lens[i]
        out_list.append('\n'.join(pred_str_list))
    
    return '\n'.join(out_list)

def compare_targets(label_path, pred_path):
    if (pred_path is None or label_path is None):
        print('Invalid file name.')
        return

    lines1 = codecs.open(label_path, mode='r', encoding='utf-8').readlines()
    lines2 = codecs.open(pred_path, mode='r', encoding='utf-8').readlines()

    length1 = len(lines1)
    length2 = len(lines2)

    if (length1 != length2):
        print('Number of lines is not same between the two files.')
        return

    errors = 0.0
    for i in range(length1):
        if (lines1[i] != lines2[i]):
            errors += 1.0
    res = errors / length1

    print('Precision: %.2f%%' % ((1.0 - res) * 100))
    return (1.0 - res)

def check_result(label_path, pred_path, unseen, exluce_array=False):
    if (pred_path is None or label_path is None):
        print("error: path is none.")
        return

    lines1 = codecs.open(label_path, encoding='utf-8').readlines()
    lines2 = codecs.open(pred_path, encoding='utf-8').readlines()

    length1 = len(lines1)
    length2 = len(lines2)

    if (length1 != length2):
        print('Number of lines is not same between the two files.')
        return

    tp = 0
    fp = 0
    fn = 0
    utp = 0
    for idx, (l1, l2) in enumerate(zip(lines1, lines2)):
        splits1 = l1.strip().split('\t')
        splits2 = l2.strip().split('\t')
        splits1 = splits1[:10]
        splits2 = splits2[:10]
        if((len(splits1) == 1 and splits1[0] != 'O') or len(splits1) > 1):
            if((len(splits2) == 1 and splits2[0] != 'O') or len(splits2) > 1):
                if(exluce_array):
                    splits1 = ['*' if s == 'array' else s for s in splits1]
                    splits2 = ['*' if s == 'array' else s for s in splits2]
                    if (splits1 != splits2):
                        fp += 1.0
                    else:
                        tp += 1.0
                else:
                    if (splits1 != splits2):
                        fp += 1.0
                    else:
                        tp += 1.0
                        if idx in unseen:
                            utp += 1.0
            else:
                fn += 1.0

    res = tp / (tp + fp)
    ures = utp / len(unseen)
    # print('Precision: %.2f%%' % ((tp / (tp + fp)) * 100))
    # print('Unseen Precision: %.2f%%' % ((utp / len(unseen)) * 100))
    # print('Recall: %.2f%%' % ((tp / (tp + fn)) * 100))
    
    return res, ures

if __name__ == '__main__':
    # DATA_PATH = './'
    # LABEL_FILE_INFER = os.path.join(DATA_PATH, 'test.lf.data')
    # NER_FILE_INFER = os.path.join(DATA_PATH, 'models/best/best.test.infer.ner')

    # check_result(LABEL_FILE_INFER, NER_FILE_INFER)
    # compare_targets(LABEL_FILE_INFER, NER_FILE_INFER)
    s = "(*(struct(*(int),*(int),*(long),*(char),int,int,int,int,long,long,long,long,*(int),*(int),*(long),*(char))),*(*(struct(enum,*(char),*((*`,long,long,int,*`,*`)->(int)),int,*((*`,long)->(int)),*((*`,*`)->(int)),*((*`,*`)->(int)),*((*`,*`,*`)->(int)),*((*`,int)->(int)),*((*`,int)->(int)),*((*`)->(void)),*((*`,*`)->(int)),*((*`,*`,long,*`)->(int)),*((*`,int,*`,*`)->(int)),*((*`,*`)->(int)),*((*`,int,*`)->(int))))),*(*(struct(int,*(struct`),long,long,long,long,*(char),int,long,long,long,int,int,int,*(void),long,*(struct`),*(*`),long,*(void)))))->(int)"
    proc_label(s)