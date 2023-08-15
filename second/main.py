# coding=utf-8
import os
import math
import time
import datetime
import codecs as co
from shutil import copy
from copy import deepcopy
from pathlib import PurePath

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import second.args as args
from second.model import TransformerModel
import second.data_utils as du
from second.graph import gen_not_seen

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#root path
DATA_PATH = './'
#check ner path
LABEL_FILE_INFER = str(PurePath(DATA_PATH) / 'test.lf.data_')
NER_FILE_INFER = str(PurePath(DATA_PATH) / 'test.infer.ner')
TEST_FILE_INFER = str(PurePath(DATA_PATH) / 'test.test.infer.ner')
# vocabulary files
WORD_VOCAB_FILE = str(PurePath(DATA_PATH) / PurePath(args.WORD_VOC))
LABEL_VOCAB_FILE = str(PurePath(DATA_PATH) / PurePath(args.NER_LABEL))
# recode result
INFER_PRECISION = './models/infer_precision'
LOSS_RECORD = './models/loss_record'
# model path
SAVE_PATH = './model.pt'
# unseen 
UNSEEN_FILE = './ner_data/unseen.data'

# src_vocab, tgt_vocab, tgt_vocab_rev = du.gen_vocab(WORD_VOCAB_FILE, LABEL_VOCAB_FILE)
src_vocab, tgt_vocab, tgt_vocab_rev = None, None, None
# _, ulist = gen_not_seen(args.MAIN_PATH+'complex/', args.MAIN_PATH+'complex/test/')
_, ulist = None, None
# ulist = du.read_unseen_file(UNSEEN_FILE)

def train(params):
    epoch_avg_loss = []
    infer_acc = 0.0
    unseen_acc = 0.0
    
    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)
    ckpt_path = os.path.join(params.out_dir,'model.pth')
    best_path = str(PurePath(params.out_dir) / 'best/')
    if not os.path.exists(best_path):
        os.makedirs(best_path)
    ubest_path = str(PurePath(params.out_dir) / 'ubest/')
    if not os.path.exists(ubest_path):
        os.makedirs(ubest_path)

    seqs, cases, labels, seq_num, max_count, batch_length, batch_size = du.get_data(str(PurePath(args.MAIN_PATH) / 'simple/'), str(PurePath(args.MAIN_PATH) / 'complex/'), params.read_data_len, mode="train")
    seqs_list, cases_list, tags_list = du.process_train_data(seqs, cases, labels, max_count, params.num_tags, src_vocab, tgt_vocab)
    assert len(seqs_list) == len(cases_list) == len(tags_list) == len(batch_size)

    model = TransformerModel(params.src_vocab_size, params.tgt_vocab_size, params.embedding_size, params.num_heads,
                 params.num_units, params.num_layers, params.num_tags, params.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    if(params.opttype == "Adam"):
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate, weight_decay=params.l2_rate)
    else:
        raise ValueError('Unsupported argument for the optimizer')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: params.decay_rate**epoch)
    
    for epoch in range(params.epochs):
        for idx, (seq, case, tag) in enumerate(zip(deepcopy(seqs_list), deepcopy(cases_list), deepcopy(tags_list))):
            train_dataset = du.MyDataset(seq, case, tag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size[idx], collate_fn=du.my_collate)
            seq_lens = seq_num[idx]

            with tqdm(train_loader, ncols=80) as pbar:
                for i, batch in enumerate(pbar):
                    batch_tokens = batch['token']
                    batch_cases = batch['case']
                    batch_labels = batch['label']
                
                    loss = model(batch_tokens, batch_cases, batch_labels)
                    avg_loss = loss / len(batch_tokens)
                    epoch_avg_loss.append(avg_loss.detach().cpu().numpy())

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    pbar.set_description('loss:%f, epoch:%d' % (loss, epoch))
            pbar.close()

        scheduler.step()
        print("learning rate: ", scheduler.get_last_lr())

        with co.open(LOSS_RECORD, 'a+', encoding='utf-8') as loss_w:
            loss_w.write('epoch ' + str(epoch) + ' : ' + str(np.mean(epoch_avg_loss)) + '\n')
            epoch_avg_loss = []
            # save model
            torch.save(model.state_dict(), SAVE_PATH)
            infer_test, unseen_test= infer(params, 'first_inferred/test/', SAVE_PATH)
            if(infer_test >= infer_acc):
                infer_acc = infer_test
                torch.save(model.state_dict(), str(PurePath(best_path) / 'model.pt'))
                copy(NER_FILE_INFER, PurePath(best_path) / 'best.test.infer.ner')
            if(unseen_test >= unseen_acc):
                unseen_acc = unseen_test
                torch.save(model.state_dict(), str(PurePath(ubest_path) / 'model.pt'))
                copy(NER_FILE_INFER, PurePath(ubest_path) / 'best.test.infer.ner')

def infer(params, dpath, mpath, func=False):
    if func is False:
        seqs, cases, labels, seq_lens, tag_lens = du.get_data(str(PurePath(args.MAIN_PATH) / dpath), str(PurePath(args.MAIN_PATH) / 'complex/test/'), params.read_data_len, mode="test")
        LABEL_FILE_INFER = str(PurePath(DATA_PATH) / 'test.lf.data_')
    else:
        seqs, cases, labels, seq_lens, tag_lens = du.get_data(str(PurePath(args.MAIN_PATH) / dpath), str(PurePath(args.MAIN_PATH) / 'complex/func/'), params.read_data_len, mode="test")
        LABEL_FILE_INFER = str(PurePath(DATA_PATH) / 'test.lf.data_func')
    seqs, cases, tags = du.process_data(seqs, cases, labels, max(seq_lens), params.num_tags, src_vocab, tgt_vocab) 

    model = TransformerModel(params.src_vocab_size, params.tgt_vocab_size, params.embedding_size, params.num_heads,
                 params.num_units, params.num_layers, params.num_tags, params.dropout)
    # model.cuda()
    if torch.cuda.is_available():
        model.cuda(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(mpath, map_location=lambda storage, loc: storage.cuda(0)))
    model.load_state_dict(torch.load(mpath, map_location=device))
    if torch.cuda.is_available():
        model.to(device)
    model.eval()

    test_dataset = du.MyDataset(seqs, cases, tags)
    test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch_size, collate_fn=du.my_collate)

    res_list = []
    with tqdm(test_loader, ncols=80) as pbar:
        for i, batch in enumerate(pbar):
            batch_tokens = batch['token']
            batch_cases = batch['case']
            batch_labels = batch['label']

            spos = i * params.batch_size
            epos = (i+1) * params.batch_size

            pred, _ = model(batch_tokens, batch_cases, batch_labels)
            res_list.append(du.process_result(pred, seq_lens[spos:epos], tag_lens[spos:epos], tgt_vocab_rev))

    res_str = '\n'.join(res_list)
    with co.open(NER_FILE_INFER, mode='w', encoding='utf-8') as infer_f:
        infer_f.write(res_str)
        infer_f.flush()
    
    # precision = du.compare_targets(NER_FILE_INFER, LABEL_FILE_INFER)
    precision, uprecision = du.check_result(LABEL_FILE_INFER, NER_FILE_INFER, ulist)
    with co.open(INFER_PRECISION, mode='a+', encoding='utf-8') as infer_w:
        infer_w.write(str(precision) + '\t' + str(uprecision) + '\n')
        infer_w.flush()
    
    return precision, uprecision


if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    s = time.time()
    train(args.params)
    infer(args.params, 'simple/test/', './models/ubest/model.pt')
    e = time.time()
    print('It took {} to finish training.'.format(datetime.timedelta(seconds=e-s)))
    print('On average, an epoch takes {}.'.format(datetime.timedelta(seconds=(e-s) / args.params.epochs)))