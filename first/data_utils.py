# coding=utf-8
import os
import codecs
from pathlib import PurePath

from torch.utils.data import Dataset
from sklearn.metrics import precision_score

import first.args as args

NMT_UNK_ID = 0
NMT_SOS_ID = 1
NMT_EOS_ID = 2
NMT_PAD_ID = 3

def sorted_nt(data_list, key=str.upper, reverse=False):
    keys = {
        '!': 0, '#': 1, '$': 2, '%': 3, '&': 4, '(': 5, ')': 6, '+': 7, ',': 8, '-': 9, '.': 10, '0': 11, '1': 12, '2': 13, '3': 14, '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20, ';': 21, '=': 22, '@': 23, 'A': 24, 'B': 25, 'C': 26, 'D': 27, 'E': 28, 'F': 29, 'G': 30, 'H': 31, 'I': 32, 'J': 33, 'K': 34, 'L': 35, 'M': 36, 'N': 37, 'O': 38, 'P': 39, 'Q': 40, 'R': 41, 'S': 42, 'T': 43, 'U': 44, 'V': 45, 'W': 46, 'X': 47, 'Y': 48, 'Z': 49, '[': 50, ']': 51, '^': 52, '_': 53, '`': 54, '{': 55, '}': 56, '~': 57
    }
    return sorted(data_list, key=lambda x: [keys.get(c, ord(c)) for c in key(x)], reverse=reverse)

def my_collate(data_list):
    sents = []
    types = []
    for data in data_list:
        sents.append(data['token'])
        types.append(data['label'])
    return {'token': sents, 'label': types}

class MyDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
    
    def __getitem__(self, index):
        tokens = self.tokens[index]
        labels = self.labels[index]
        return {'token': tokens, 'label': labels}

    def __len__(self):
        return len(self.tokens)


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

def get_data(path, max_len, mode="test"): 
    labels = []
    seq = []
    words = []
    label = []
    counts = []
    data_list = read_data_file(path)

    if(mode == "train"):
        seq_num = []
        max_count = []
        batch_length = [50, 100, 200, 400, 600, max_len]
        # batch_size = [64, 32, 16, 8, 4, 2]
        batch_size = [128, 64, 32, 16, 8, 4]
        for i in range(len(batch_length)):
            labels.append([])
            seq.append([])
            counts.append([])
    
    for item in data_list:
        # each file
        item = item.strip().split('\n')
        if(len(item) > max_len):
            continue
        count = 0
        for data in item:
            # each line
            data = data.strip().rsplit('\t', 1)
            if(len(data) <= 1):
                continue
            count += 1
            words.append(data[0])
            label.append(data[1])
            assert len(words) == len(label)
            if(len(words) == max_len):
                break
        
        if(len(words) > 0):
            if(mode == "train"):
                i = [idx for idx in range(len(batch_length)) if batch_length[idx] >= len(words)][0]
                seq[i].append(words)
                labels[i].append(label)
                counts[i].append(count)
            else:
                seq.append(words)
                labels.append(label)
                counts.append(count)
        words = []
        label = []
    
    if(mode == "train"):
        for i in range(len(batch_length)):
            seq_num.append(len(seq[i]))
            max_count.append(max(counts[i]))
    
        # seq           根据句子长度区分的所有token序列
        # labels        根据句子长度区分的所有标签序列
        # seq_num       每个batch区间内部句子的个数
        # max_count     每个batch区间内句子长度最大值
        # batch_length  每个batch区间句子的最大长度
        # batch_size    每次训练batch个数
        return seq, labels, seq_num, max_count, batch_length, batch_size
    else:
        # seq           所有token序列
        # labels        所有标签序列
        # max(counts)   句子长度
        return seq, labels, counts

def gen_vocab(src_vocab_file, tgt_vocab_file):
    src_vocab = read_vocab_file(src_vocab_file)
    tgt_vocab = read_vocab_file(tgt_vocab_file)
    tgt_vocab_rev = {v: k for k, v in tgt_vocab.items()}

    return src_vocab, tgt_vocab, tgt_vocab_rev

def process_train_data(seqs_list, tags_list, max_lens, src_vocab, tgt_vocab):
    assert len(seqs_list) == len(tags_list) == len(max_lens)
    for i, (seqs, tags, max_len) in enumerate(zip(seqs_list, tags_list, max_lens)):
        # 每个长度区间
        seqs, tags = process_data(seqs, tags, max_len, src_vocab, tgt_vocab)
        # seqs_list[i] = seqs
        # tags_list[i] = tags
    
    return seqs_list, tags_list

def process_data(seqs, tags, max_len, src_vocab, tgt_vocab):
    assert len(seqs) == len(tags)
    for i, (seq, tag) in enumerate(zip(seqs, tags)):
        # 每个文件
        assert len(seq) == len(tag)
        for idx, (token, label) in enumerate(zip(seq, tag)):
            # 文件内每行
            seq[idx] = src_vocab[token] if token in src_vocab else 0
            tag[idx] = tgt_vocab[label] if label in tgt_vocab else 0
        seq += [NMT_EOS_ID]
        tag += [NMT_EOS_ID]
        assert len(seq) == len(tag)
        if(len(seq) < max_len+1):
            seq += [NMT_PAD_ID] * (max_len-len(seq)+1)
            tag += [NMT_PAD_ID] * (max_len-len(tag)+1)
    
    # still need torch embedding
    return seqs, tags

def process_result(preds, seq_lens, tgt_vocab_rev):
    # each batch
    out_list = []
    for i, pred in enumerate(preds):
        # each file
        pred_str_list = []
        for tag in pred:
            # each pred tag
            pred_str_list.append(tgt_vocab_rev[tag])
        
        pred_str_list = pred_str_list[:seq_lens[i]]
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

    # print('Precision: %.2f%%' % ((1.0 - res) * 100))
    return (1.0 - res)

def check_result(label_path, pred_path, exluce_array=False):
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
    for i in range(length1):
        if (lines1[i] != 'O'):
            if(lines2[i] != 'O'):
                if(exluce_array):
                    if (('array' in lines1[i] and 'pointer' in lines2[i]) or ('pointer' in lines1[i] and 'array' in lines2[i])):
                        tp += 1.0
                    elif (lines1[i].strip() != lines2[i].strip()):
                        fp += 1.0
                    else:
                        tp += 1.0
                else:
                    if (lines1[i].strip() != lines2[i].strip()):
                        fp += 1.0
                    else:
                        tp += 1.0
            else:
                fn += 1.0

    res = tp / (tp + fp)
    rec = tp / (tp + fn)
    print('    Precision: %.2f%%' % ((tp / (tp + fp)) * 100))
    print('    Recall: %.2f%%' % ((tp / (tp + fn)) * 100))
    print('    F1: %.2f%%' % ((2 * res * rec / (res + rec)) * 100))
    
    return res


def check_result_multi_class(label_path, pred_path, vocab_path, mode="all"):
    if (pred_path is None or label_path is None):
        print("error: path is none.")
        return
    
    vocab = read_vocab_file(vocab_path)
    # exclude array
    for k, v in vocab.items():
            if v == 9:
                vocab[k] = 5
    
    if mode == "all":
        include = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    elif mode == "simple":
        include = [4, 6, 8, 10, 13, 14, 16, 17, 18, 19, 20]
    elif mode == "complex":
        include = [5, 7, 9, 11, 12, 15]
    else:
        print("wrong parameter")
        return

    lines1 = codecs.open(label_path, encoding='utf-8').readlines()
    lines2 = codecs.open(pred_path, encoding='utf-8').readlines()

    length1 = len(lines1)
    length2 = len(lines2)

    if (length1 != length2):
        print('Number of lines is not same between the two files.')
        return

    true = [vocab[l.strip()] for l in lines1]
    pred = [vocab[l.strip()] for l in lines2]

    res = precision_score(true, pred, average='micro', labels=include, zero_division=1)
    print('    Accuracy: %.2f%%' % (res * 100))
    
    return res


if __name__ == '__main__':
    DATA_PATH = './'
    LABEL_FILE_INFER = os.path.join(DATA_PATH, 'test.lf.data')
    NER_FILE_INFER = os.path.join(DATA_PATH, 'models_deeptyper_consis/best/best.test.infer.ner')

    check_result(LABEL_FILE_INFER, NER_FILE_INFER)
    check_result(LABEL_FILE_INFER, NER_FILE_INFER, exluce_array=True)
    # compare_targets(LABEL_FILE_INFER, NER_FILE_INFER)