# encoding=utf-8
import os
import codecs
from pathlib import PurePath

import args
from data_utils import proc_label
from graph import gen_not_seen

label_path = PurePath('./') / PurePath('test.lf.data_')
pred_path = PurePath('./models/ubest/') / PurePath('best.test.infer.ner')
# pred_path = PurePath('./') / PurePath('test.infer.ner')
write_path = PurePath('./models/ubest/') / PurePath('different.data')

infer_path = PurePath('./ner_data/') / PurePath('infer_first.data')
first_path = PurePath('./ner_data/') / PurePath('infer/test/')
write_first_path = PurePath('./ner_data/') / PurePath('infer/test_infer/')

train_dir = PurePath('./ner_data/') / PurePath('complex/')
test_dir = PurePath('./ner_data/') / PurePath('complex/test/')
ulist = list()

def get_different(l_path, p_path, w_path):
    tp = 0
    fp = 0
    fn = 0
    utp = 0
    unum = 0

    with codecs.open(w_path, mode='w', encoding='utf-8') as file_w:
        lines1 = codecs.open(l_path, encoding='utf-8').readlines()
        lines2 = codecs.open(p_path, encoding='utf-8').readlines()
        assert len(lines1) == len(lines2)
        for idx, (l1, l2) in enumerate(zip(lines1, lines2)):
            splits1 = l1.strip().split('\t')
            splits2 = l2.strip().split('\t')
            splits1 = splits1[:10]
            splits2 = splits2[:10]
            if((len(splits1) == 1 and splits1[0] != 'O') or len(splits1) > 1):
                if((len(splits2) == 1 and splits2[0] != 'O') or len(splits2) > 1):
                    if(splits1 != splits2):
                        if(idx in ulist):
                            file_w.write('unseen\n')
                        file_w.write(str(idx) + '\n' + '\t'.join(splits1) + '\n' + '\t'.join(splits2) + '\n\n')
                        fp += 1.0
                    else:
                        tp += 1.0
                        if(idx in ulist):
                            utp += 1.0
                else:
                    fn += 1.0
    print('tp: %f, fp: %f, fn: %f' % (tp, fp, fn))
    print('Precision: %.2f%%' % ((tp / (tp + fp)) * 100))
    print('Recall: %.2f%%' % ((tp / (tp + fn)) * 100))
    print('utp: %f, useen num: %f' % (utp, len(ulist)))
    print('Unseen: %.2f%%' % ((utp / len(ulist)) * 100))

def read_data_file(path):
    data_list = []
    with open(path, 'r') as file_r:
        data_f = file_r.read().strip().split('\n')
        for data in data_f:
            splits = data.rsplit('\t', 1)
            if(len(splits) <= 1):
                continue
            data_list.append(splits[1])
    
    return data_list

def gen_infer_first(p_path, t_dir, w_dir):
    first_list = read_data_file(p_path)

    for item in os.listdir(t_dir):
        str_list = []
        name = t_dir + item
        if(item[0] != '.' and os.path.isfile(name)): 
            with open(name, 'r') as file_r:
                data_f = file_r.read().strip()
                lines = data_f.strip().split('\n')
                if(len(lines) > args.params.read_data_len):
                    continue
                for data in lines:
                    # each line
                    splits = data.strip().rsplit('\t', 1)
                    if(len(splits) <= 1):
                        continue
                    str_list.append(splits[0] + "\t" + first_list.pop(0))
            
            with open(name, mode='w', encoding='utf-8') as write_f:
                write_f.write('\n'.join(str_list))
                write_f.flush()

    assert len(first_list) == 0, "error!"


if __name__ == '__main__':
    _, ulist = gen_not_seen(train_dir, test_dir)
    get_different(label_path, pred_path, write_path)
    # gen_infer_first(infer_path, first_path, write_first_path)
