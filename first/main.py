# coding=utf-8
import os
import math
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

import first.args as args
from first.model import Model
import first.data_utils as du

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#root path
DATA_PATH = './'
#check ner path
LABEL_FILE_INFER = str(PurePath(DATA_PATH) / 'test.lf.data')
NER_FILE_INFER = str(PurePath(DATA_PATH) / 'test.infer.ner')
# vocabulary files
WORD_VOCAB_FILE = str(PurePath(DATA_PATH) / PurePath(args.WORD_VOC))
LABEL_VOCAB_FILE = str(PurePath(DATA_PATH) / PurePath(args.NER_LABEL))
# recode result
INFER_PRECISION = './models/infer_precision'
LOSS_RECORD = './models/loss_record'
# model path
SAVE_PATH = './model.pt'

# src_vocab, tgt_vocab, tgt_vocab_rev = du.gen_vocab(WORD_VOCAB_FILE, LABEL_VOCAB_FILE)
src_vocab, tgt_vocab, tgt_vocab_rev = None, None, None

def train(params):
    epoch_avg_loss = []
    infer_acc = 0.0
    
    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)
    ckpt_path = os.path.join(params.out_dir,'model.pth')
    best_path = str(PurePath(params.out_dir) / 'best/')
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    seqs, labels, seq_num, max_count, batch_length, batch_size = du.get_data(str(PurePath(args.MAIN_PATH) / 'simple/'), params.read_data_len, mode="train")
    seqs_list, tags_list = du.process_train_data(seqs, labels, max_count, src_vocab, tgt_vocab)
    assert len(seqs_list) == len(tags_list) == len(batch_size)

    model = Model(params.src_vocab_size, params.tgt_vocab_size, params.embedding_size, params.mlp_hidden,
                 params.num_units, params.unit_type, params.dropout, params.batch_first, params.need_atten)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to('cuda:0')
    model.to(device)
    model.train()
    
    if(params.opttype == "Adam"):
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate, weight_decay=params.l2_rate)
    else:
        raise ValueError('Unsupported argument for the optimizer')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: params.decay_rate**epoch)
    
    for epoch in range(params.epochs):
        for idx, (seq, tag) in enumerate(zip(deepcopy(seqs_list), deepcopy(tags_list))):
            train_dataset = du.MyDataset(seq, tag)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size[idx], collate_fn=du.my_collate)

            with tqdm(train_loader, ncols=80) as pbar:
                for i, batch in enumerate(pbar):
                    batch_tokens = batch['token']
                    batch_labels = batch['label']
                
                    loss = model(batch_tokens, tags=batch_labels)
                    loss = loss / len(batch_tokens)
                    epoch_avg_loss.append(loss.detach().cpu().numpy())

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
            infer_test = infer(params)
            if(infer_test >= infer_acc):
                infer_acc = infer_test
                torch.save(model.state_dict(), str(PurePath(best_path) / 'model.pt'))
                copy(NER_FILE_INFER, PurePath(best_path) / 'best.test.infer.ner')

def infer(params, func=False):
    if func is False:
        seqs, labels, seq_lens = du.get_data(str(PurePath(args.MAIN_PATH) / 'simple/test/'), params.read_data_len, mode="test")
        LABEL_FILE_INFER = str(PurePath(DATA_PATH) / 'test.lf.data')
    else:
        seqs, labels, seq_lens = du.get_data(str(PurePath(args.MAIN_PATH) / 'simple/func/'), params.read_data_len, mode="test")
        LABEL_FILE_INFER = str(PurePath(DATA_PATH) / 'test.lf.data_func')
    for idx, (seq, l) in enumerate(zip(seqs, seq_lens)):
        try:
            assert len(seq) == l
        except:
            print("index: ", idx)
            print("len of seq: ", len(seq))
            print("len of mask: ", l)
    seqs, tags = du.process_data(seqs, labels, max(seq_lens), src_vocab, tgt_vocab) 

    model = Model(params.src_vocab_size, params.tgt_vocab_size, params.embedding_size, params.mlp_hidden,
                 params.num_units, params.unit_type, params.dropout, params.batch_first, params.need_atten)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.cuda()
    if torch.cuda.is_available():
        model.cuda()
    # model.load_state_dict(torch.load(SAVE_PATH, map_location=lambda storage, loc: storage.cuda(0)))
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    if torch.cuda.is_available():
        model.to(device)
    model.eval()
    # torch.save(model.state_dict(), './model.pt')
    test_dataset = du.MyDataset(seqs, tags)
    test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch_size, collate_fn=du.my_collate)

    res_list = []
    with tqdm(test_loader, ncols=80) as pbar:
        for i, batch in enumerate(pbar):
            batch_tokens = batch['token']
            # batch_labels = batch['label']
            spos = i * params.batch_size
            epos = (i+1) * params.batch_size

            pred = model(batch_tokens)
            res_list.append(du.process_result(pred, seq_lens[spos:epos], tgt_vocab_rev))

    res_str = '\n'.join(res_list)
    with co.open(NER_FILE_INFER, mode='w', encoding='utf-8') as infer_f:
        infer_f.write(res_str)
        infer_f.flush()
    
    precision = du.compare_targets(NER_FILE_INFER, LABEL_FILE_INFER)
    # precision = du.check_result(NER_FILE_INFER, LABEL_FILE_INFER)
    with co.open(INFER_PRECISION, mode='a+', encoding='utf-8') as infer_w:
        infer_w.write(str(precision) + '\n')
        infer_w.flush()
    
    return precision


if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    train(args.params)