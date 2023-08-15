#!/usr/bin/env python
import argparse
import linecache
import os
import codecs
import hashlib
from pathlib import PurePath

import torch
import torch.cuda
import numpy as np

import first.args
import first.data_utils
import first.main
from first.data_utils import sorted_nt

import second.args
import second.data_utils
import second.main
import second.graph
import second.transfer

def get_dir_hash(path):
    hasher = hashlib.sha256()
    for file in sorted_nt(os.listdir(path)):
        filepath = PurePath(path) / PurePath(file)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
            except Exception as e:
                print(str(e))
    return hasher.hexdigest()

def get_hash(dir_list, file_list):
    hasher = hashlib.sha256()
    for dir in dir_list:
        hasher.update(get_dir_hash(dir).encode('utf-8'))
    for file in file_list:
        with open(file, 'rb') as f:
            hasher.update(f.read())
    return hasher.hexdigest()

class FirstStage:
    @staticmethod
    def vocab(mode: int, data_path: str, out_path: str):
        # param: mode
        # 0 for word, 1 for label


        vocab_path = first.args.WORD_VOC if mode == 0 else first.args.NER_LABEL
        vocab_path = str(PurePath(out_path) / PurePath(vocab_path))
        train_path = str(PurePath(data_path) / 'simple/')
        test_path = str(PurePath(data_path) / 'simple/test/')
        hash_dir = str(PurePath(out_path) / f'vocab_{["word", "label"][mode]}_hash.txt')
        if not os.path.exists(hash_dir):
            with open(hash_dir, 'w') as file_w:
                file_w.write('')
        else:
            with open(hash_dir, 'r') as file_r:
                hash = file_r.read().strip()
                if get_hash([train_path, test_path], [vocab_path]) == hash:
                    print(f"Data not changed, skip building vocab file for {['word', 'label'][mode]}...")
                    return
        print(f"Building vocab file for {['word', 'label'][mode]}...")

        vocab_set = set()

        def change_file_code(files_name):
            try:
                cache_data = linecache.getlines(files_name)
                with open(files_name, 'wb') as out_file:
                    for line in range(len(cache_data)):
                        out_file.write(cache_data[line].encode('utf-8'))
            except Exception as e:
                print(str(e))

        def scan_file(path):
            for item in sorted_nt(os.listdir(path)):
                name = str(PurePath(path) / item)
                temp_set = set()
                line_num = 0
                if os.path.isfile(name):
                    # change_file_code(name)
                    try:
                        file_r = open(name, 'r', encoding='utf-8')
                        for line in file_r:
                            splits = line.strip().rsplit('\t', 1)
                            if(len(splits) <= 1):
                                continue
                            temp_set.add(splits[mode])
                            line_num += 1
                    except:
                        print(name)
                    
                    if(line_num <= first.args.params.read_data_len):
                        vocab_set.update(temp_set)
        
        scan_file(train_path)
        scan_file(test_path)
        
        vocab_str = '<unk>\n<s>\n</s>\n<padding>\n'
        for v in vocab_set:
            vocab_str += v + '\n'
        
        with codecs.open(str(vocab_path), 'w', encoding='utf-8') as file_w:
            file_w.write(vocab_str)

        with open(hash_dir, 'w') as file_w:
            file_w.write(get_hash([train_path, test_path], [vocab_path]))
    
    @staticmethod
    def target(data_path: str, out_path: str):
        target_file_path = str(PurePath(out_path) / 'test.lf.data')
        dir_path = PurePath(data_path) / 'simple/test/'
        hash_dir = str(PurePath(out_path) / 'target_hash.txt')
        if not os.path.exists(hash_dir):
            with open(hash_dir, 'w') as file_w:
                file_w.write('')
        else:
            with open(hash_dir, 'r') as file_r:
                hash = file_r.read().strip()
                if get_hash([str(dir_path)], [target_file_path]) == hash:
                    print("Data not changed, skip building target file...")
                    return
        print("Building target file for test...")
        with codecs.open(target_file_path, mode='w', encoding='utf-8') as file_w:
            items = sorted_nt(os.listdir(dir_path))
            for item in items:
                file = dir_path.joinpath(item)
                lines = []
                line_num = 0
                for line in open(file, 'r', encoding='utf-8'):
                    splits = line.strip().rsplit('\t', 1)
                    lines.append(splits[1]+'\n')
                    line_num += 1
                
                if(line_num <= first.args.params.read_data_len):
                    file_w.writelines(lines)
            
            file_w.close()

        with open(hash_dir, 'w') as file_w:
            file_w.write(get_hash([str(dir_path)], [target_file_path]))

    @staticmethod
    def train(data_path: str):
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        first.args.MAIN_PATH = data_path
        first.main.DATA_PATH = './models/first/out/'
        first.main.LABEL_FILE_INFER = str(PurePath(first.main.DATA_PATH) / 'test.lf.data')
        first.main.NER_FILE_INFER = str(PurePath(first.main.DATA_PATH) / 'test.infer.ner')
        first.main.WORD_VOCAB_FILE = str(PurePath('./models/first/out/') / PurePath(first.args.WORD_VOC))
        first.main.LABEL_VOCAB_FILE = str(PurePath('./models/first/out/') / PurePath(first.args.NER_LABEL))
        first.main.INFER_PRECISION = './models/first/infer_precision'
        first.main.LOSS_RECORD = './models/first/loss_record'
        first.main.SAVE_PATH = str(PurePath('./models/first/') / 'model.pt')
        first.main.src_vocab, first.main.tgt_vocab, first.main.tgt_vocab_rev = first.data_utils.gen_vocab(first.main.WORD_VOCAB_FILE, first.main.LABEL_VOCAB_FILE)
        first.args.params.out_dir = './models/first/'
        print("Begin Training:")
        first.main.train(first.args.params)

    @staticmethod
    def eval(data_path: str, models):
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        # if model == 'ALL':
        #     models = ['TRAINED', 'STIR', 'STIR_A', 'DeepTyper']
        # elif model == 'PRETRAINED':
        #     models = ['STIR', 'STIR_A', 'DeepTyper']
        # else:
        #     models = [model]
        
        model_path = {
            'TRAINED': './models/first/',
            'STIR': './pretrained/first/stir/',
            'STIR_A': './pretrained/first/stir_a/',
            'DeepTyper': './pretrained/first/deeptyper/'
        }

        model_batch_size = {
            'TRAINED': 64,
            'STIR': 64,
            'STIR_A': 8,
            'DeepTyper': 8 
        }

        for model in models:
            if model == 'TRAINED' and not os.path.exists('./models/first/model.pt'):
                print("Self-trained model not found, please train a model before evaluating...")
                continue
            print("Evaluating model: " + model)
            first.args.MAIN_PATH = data_path
            # first.args.WORD_VOC = str(PurePath(model_path[model]) / 'out/vocab.na.data')
            # first.args.NER_LABEL = str(PurePath(model_path[model]) / 'out/vocab.lf.data')
            first.main.DATA_PATH = str(PurePath(model_path[model]) / 'out/')
            first.main.LABEL_FILE_INFER = str(PurePath(first.main.DATA_PATH) / 'test.lf.data')
            first.main.NER_FILE_INFER = str(PurePath(first.main.DATA_PATH) / 'test.infer.ner')
            first.main.WORD_VOCAB_FILE = str(PurePath(model_path[model]) / 'out' / first.args.WORD_VOC)
            first.main.LABEL_VOCAB_FILE = str(PurePath(model_path[model]) / 'out' / first.args.NER_LABEL)
            first.main.INFER_PRECISION = str(PurePath(model_path[model]) / 'infer_precision')
            first.main.LOSS_RECORD = str(PurePath(model_path[model]) / 'loss_record')
            first.main.SAVE_PATH = str(PurePath(model_path[model]) / 'best/model.pt')
            first.main.src_vocab, first.main.tgt_vocab, first.main.tgt_vocab_rev = first.data_utils.gen_vocab(first.main.WORD_VOCAB_FILE, first.main.LABEL_VOCAB_FILE)
            first.args.params.batch_size = model_batch_size[model]
            first.args.params.need_atten = model != 'STIR'
            first.args.params.unit_type = first.args.RNN_UNIT_TYPE_GRU if model == 'DeepTyper' else first.args.RNN_UNIT_TYPE_LSTM
            print("Begin Inference:")
            first.main.infer(first.args.params)

            label_path = str(PurePath(model_path[model]) / 'out/test.lf.data')
            infer_path = str(PurePath(model_path[model]) / 'out/test.infer.ner')
            vocab_path = str(PurePath(model_path[model]) / 'out/vocab.lf.data')
            print("Processsing Result...")
            print("SimpleType:")
            first.data_utils.check_result_multi_class(label_path, infer_path, vocab_path, mode='simple')
            print("ComplexType:")
            first.data_utils.check_result_multi_class(label_path, infer_path, vocab_path, mode='complex')
            print("AllType:")
            first.data_utils.check_result_multi_class(label_path, infer_path, vocab_path, mode='all')
    
    @staticmethod
    def gen_data_for_second(data_path: str, model: str):
        model_path = {
            'TRAINED': './models/first/',
            'STIR': './pretrained/first/stir/',
            'STIR_A': './pretrained/first/stir_a/',
            'DeepTyper': './pretrained/first/deeptyper/'
        }
        
        # infer_best_path = str(PurePath(model_path[model]) / 'best/best.test.infer.ner')
        # if not (os.path.exists(infer_best_path) and os.path.isfile(infer_best_path)):
            # infer_best_path = str(PurePath(model_path[model]) / 'out/test.infer.ner')
        infer_best_path = str(PurePath(model_path[model]) / 'out/test.infer.ner')
        first_data_path = str(PurePath(data_path) / 'simple/test/')
        infer_data_path = str(PurePath(data_path) / 'first_inferred/test/')

        inferred_result_list = []
        if os.path.isfile(infer_best_path):
            with open(infer_best_path, 'r') as file_r:
                inferred_result_list = file_r.read().strip().split('\n')

        for item in sorted_nt(os.listdir(first_data_path)):
            str_list = []
            first_data_file = str(PurePath(first_data_path) / item)
            if(item[0] != '.' and os.path.isfile(first_data_file)):
                with open(first_data_file, 'r') as file_r:
                    first_data_file_content = file_r.read().strip()
                lines = first_data_file_content.strip().split('\n')
                if(len(lines) > first.args.params.read_data_len):
                    with open(str(PurePath(infer_data_path) / item), mode='w', encoding='utf-8') as write_f:
                        write_f.write(first_data_file_content)
                        write_f.flush()
                    continue
                for line in lines:
                    # each line
                    splits = line .strip().rsplit('\t', 1)
                    if(len(splits) <= 1):
                        continue
                    inferred_result = inferred_result_list.pop(0)
                    if inferred_result == '<padding>':
                        inferred_result = 'pointer'
                    str_list.append(splits[0] + "\t" + inferred_result)
                with open(str(PurePath(infer_data_path) / item), mode='w', encoding='utf-8') as write_f:
                    write_f.write('\n'.join(str_list))
                    write_f.flush()
        
        assert len(inferred_result_list) == 0, "error!"
        
class SecondStage:
    @staticmethod
    def vocab(mode: int, data_path: str, out_path: str):
        # param: mode
        # 0 for word, 1 for label
        vocab_path = second.args.WORD_VOC if mode == 0 else second.args.NER_LABEL
        vocab_path = str(PurePath(out_path) / PurePath(vocab_path))
        train_path = str(PurePath(data_path) / 'complex/')
        test_path = str(PurePath(data_path) / 'complex/test/')
        hash_dir = str(PurePath(out_path) / f'vocab_{["word", "label"][mode]}_hash.txt')

        if not os.path.exists(hash_dir):
            with open(hash_dir, 'w') as file_w:
                file_w.write('')
        else:
            with open(hash_dir, 'r') as file_r:
                hash = file_r.read().strip()
                getted_hash = get_hash([train_path, test_path], [vocab_path])
                if getted_hash == hash:
                    print("Data not changed, skip building vocab file for " + ['word', 'label'][mode] + "...")
                    return
        print("Building vocab file for " + ['word', 'label'][mode] + "...")

        vocab_set = set()

        def change_file_code(files_name):
            try:
                cache_data = linecache.getlines(files_name)
                with open(files_name, 'wb') as out_file:
                    for line in range(len(cache_data)):
                        out_file.write(cache_data[line].encode('utf-8'))
            except Exception as e:
                print(str(e))

        def scan_file(path):
            for item in sorted_nt(os.listdir(path)):
                name = str(PurePath(path) / item)
                temp_set = set()
                line_num = 0
                if(os.path.isfile(name)): 
                    # change_file_code(name)
                    try:
                        file_r = open(name, 'r', encoding='utf-8')
                        for line in file_r:
                            splits = line.strip().rsplit('\t', 1)
                            if(len(splits) <= 1):
                                continue
                            if mode == 1:
                                processed = []
                                for item in second.data_utils.proc_label(splits[1]):
                                    processed.append(second.data_utils.proc_casing(item))
                                if 'long long' in processed:
                                    print(f'name: {name}, line: {line}, splits: {splits}, processed: {processed}')
                                # temp_set.update(second.data_utils.proc_label(processed))
                                temp_set.update(processed)
                            else:
                                temp_set.add(splits[0])
                            line_num += 1
                        file_r.close()
                    except Exception as e:
                        print(e)
                        print(name)
                    
                    if(line_num <= second.args.params.read_data_len):
                        vocab_set.update(temp_set)
        
        scan_file(train_path)
        scan_file(test_path)
        
        vocab_str = '<unk>\n<padding>\n' if mode == 0 else '<padding>\n'
        for v in vocab_set:
            vocab_str += v + '\n'
        
        with codecs.open(vocab_path, 'w', encoding='utf-8') as file_w:
            file_w.write(vocab_str)

        with open(hash_dir, 'w') as file_w:
            file_w.write(get_hash([train_path, test_path], [vocab_path]))
    
    @staticmethod
    def target(data_path: str, out_path: str):
        target_file_path = str(PurePath(out_path) / 'test.lf.data_')
        dir_path = str(PurePath(data_path) / 'complex/test/')
        hash_dir = str(PurePath(out_path) / 'target_hash.txt')
        if not os.path.exists(hash_dir):
            with open(hash_dir, 'w') as file_w:
                file_w.write('')
        else:
            with open(PurePath(out_path) / 'target_hash.txt', 'r') as file_r:
                hash = file_r.read().strip()
                if get_hash([dir_path], [target_file_path]) == hash:
                    print("Data not changed, skip building target file...")
                    return
        print("Building target file for test...")
        with codecs.open(target_file_path, mode='w', encoding='utf-8') as file_w:
            items = sorted_nt(os.listdir(dir_path))
            for item in items:
                file = str(PurePath(dir_path) / item)
                lines = []
                line_num = 0
                for line in open(file, 'r', encoding='utf-8'):
                    splits = line.strip().rsplit('\t', 1)
                    lines.append('\t'.join(second.data_utils.proc_label(splits[1])) + '\n')
                    # lines.append(splits[1]+'\n')
                    line_num += 1
                
                if(line_num <= second.args.params.read_data_len):
                    file_w.writelines(lines)
            file_w.close()
        with open(hash_dir, 'w') as file_w:
            file_w.write(get_hash([dir_path], [target_file_path]))
    
    @staticmethod
    def get_result(data_path: str, model_path: str, variant=''):
        target_file_path = str(PurePath(model_path) / 'out/test.lf.data_')
        infer_file_path = str(PurePath(model_path) / f'out/test.infer.ner{variant}')
        target_file_PCFG_path = str(PurePath(model_path) / 'out/test_lf_data_PCFG.txt')
        infer_file_PCFG_path = str(PurePath(model_path) / f'out/test_infer_ner{variant}_PCFG.txt')

        pcfg_target_hash_dir = str(PurePath(model_path) / 'out/pcfg_target_hash.txt')
        pcfg_infer_hash_dir = str(PurePath(model_path) / f'out/pcfg_infer{variant}_hash.txt')
        if not os.path.exists(pcfg_target_hash_dir):
            print("Building PCFG file for target...")
            second.transfer.recover_data(target_file_path, target_file_path, target_file_PCFG_path)
            with open(pcfg_target_hash_dir, 'w') as file_w:
                file_w.write(get_hash([], [target_file_path, target_file_PCFG_path]))
        else:
            with open(pcfg_target_hash_dir, 'r') as file_r:
                hash = file_r.read().strip()
                if get_hash([], [target_file_path, target_file_PCFG_path]) == hash:
                    print("Data not changed, skip building PCFG file for target...")
                else:
                    print("Building PCFG file for target...")
                    second.transfer.recover_data(target_file_path, target_file_path, target_file_PCFG_path)
                    with open(pcfg_target_hash_dir, 'w') as file_w:
                        file_w.write(get_hash([], [target_file_path, target_file_PCFG_path]))

        if not os.path.exists(pcfg_infer_hash_dir):
            print("Building PCFG file for inferred result...")
            second.transfer.recover_data(target_file_path, infer_file_path, infer_file_PCFG_path)
            with open(pcfg_infer_hash_dir, 'w') as file_w:
                file_w.write(get_hash([], [target_file_path, infer_file_path, infer_file_PCFG_path]))
        else:
            with open(pcfg_infer_hash_dir, 'r') as file_r:
                hash = file_r.read().strip()
                if get_hash([], [target_file_path, infer_file_path, infer_file_PCFG_path]) == hash:
                    print("Data not changed, skip building PCFG file for inferred result...")
                else:
                    print("Building PCFG file for inferred result...")
                    second.transfer.recover_data(target_file_path, infer_file_path, infer_file_PCFG_path)
                    with open(pcfg_infer_hash_dir, 'w') as file_w:
                        file_w.write(get_hash([], [target_file_path, infer_file_path, infer_file_PCFG_path]))

        second.graph.trgt_path = target_file_path
        second.graph.pred_path = infer_file_path
        second.graph.trgt_path_t = target_file_PCFG_path
        second.graph.pred_path_t = infer_file_PCFG_path
        second.graph.first_trgt = str(PurePath(data_path) / 'simple/test/')
        # second.graph.first_pred = str(PurePath(data_path) / 'first_inferred/test/')
        second.graph.first_pred = str(PurePath(data_path) / 'simple/test/')
        second.graph.train_dir = str(PurePath(data_path) / 'complex/')
        second.graph.test_dir = str(PurePath(data_path) / 'complex/test/')
        print("Processsing Result, this may take a while...")
        second.graph.uset, second.graph.ulist = second.graph.gen_not_seen(second.graph.train_dir, second.graph.test_dir)
        second.graph.prim_list = list()
        second.graph.ptr_list = list()
        second.graph.stru_list = list()
        second.graph.func_list = list()
        second.graph.fw_list = list()
        second.graph.gen_first_related(second.graph.first_trgt, second.graph.first_pred)
        second.graph.toTree(second.graph.trgt_path, second.graph.pred_path, check_ot=variant == '_O')
        if variant != '_O':
            second.graph.get_type_prefix_score(second.graph.trgt_path, second.graph.pred_path, second.graph.trgt_path_t, second.graph.pred_path_t)

    @staticmethod
    def train(data_path: str):
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        second.args.MAIN_PATH = data_path
        second.main.DATA_PATH = './models/second/out/'
        second.main.LABEL_FILE_INFER = str(PurePath(second.main.DATA_PATH) / 'test.lf.data_')
        second.main.NER_FILE_INFER = str(PurePath(second.main.DATA_PATH) / 'test.infer.ner')
        second.main.WORD_VOCAB_FILE = str(PurePath('./models/second/out/') / PurePath(second.args.WORD_VOC))
        second.main.LABEL_VOCAB_FILE = str(PurePath('./models/second/out/') / PurePath(second.args.NER_LABEL))
        second.main.INFER_PRECISION = './models/second/infer_precision'
        second.main.LOSS_RECORD = './models/second/loss_record'
        second.main.SAVE_PATH = str(PurePath('./models/second') / 'model.pt')
        second.main.src_vocab, second.main.tgt_vocab, second.main.tgt_vocab_rev = second.data_utils.gen_vocab(second.main.WORD_VOCAB_FILE, second.main.LABEL_VOCAB_FILE)
        _, second.main.ulist = second.graph.gen_not_seen(str(PurePath(second.args.MAIN_PATH) / 'complex/'), str(PurePath(second.args.MAIN_PATH) / 'complex/test/'))
        second.args.params.out_dir = './models/second/'
        print("Begin Training:")
        second.main.train(second.args.params)

    @staticmethod
    def eval(data_path: str, models: str, zero_shot: bool):
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        
        model_path = {
            'TRAINED': './models/second/',
            'TRAINED_OT': './models/second/',
            'TRAINED_DT': './models/second/',
            'TRAINED_GT': './models/second/',
            'STIR': './pretrained/second/stir/',
            'STIR_OT': './pretrained/second/stir/',
            'STIR_DT': './pretrained/second/stir/',
            'STIR_GT': './pretrained/second/stir/'
        }

        data_sub_path = {
            'TRAINED': 'first_inferred/test',
            'TRAINED_OT': 'simple/test',
            'TRAINED_DT': 'deeptAten/test',
            'TRAINED_GT': 'simple/test',
            'STIR': 'first_inferred/test',
            'STIR_OT': 'simple/test',
            'STIR_DT': 'deeptAten/test',
            'STIR_GT': 'simple/test'
        }

        for model in models:
            if model[0:7] == 'TRAINED' and not os.path.exists('./models/second/ubest/model.pt' if zero_shot else './models/second/best/model.pt'):
                print("Self-trained model not found, please train a model before evaluating...")
                continue
            print("Evaluating model: " + model)
            second.args.MAIN_PATH = data_path
            second.main.DATA_PATH = str(PurePath(model_path[model]) / 'out/')
            second.main.LABEL_FILE_INFER = str(PurePath(second.main.DATA_PATH) / 'test.lf.data_')
            second.main.NER_FILE_INFER = str(PurePath(second.main.DATA_PATH) / f'test.infer.ner{model[model.find("_"):-1]}')
            second.main.WORD_VOCAB_FILE = str(PurePath(model_path[model]) / 'out' / PurePath(second.args.WORD_VOC))
            second.main.LABEL_VOCAB_FILE = str(PurePath(model_path[model]) / 'out' / PurePath(second.args.NER_LABEL))
            second.main.INFER_PRECISION = str(PurePath(model_path[model]) / 'infer_precision')
            second.main.LOSS_RECORD = str(PurePath(model_path[model]) / 'loss_record')
            second.data_utils.has_simple = False if model[-2:] == 'OT' else True
            second.main.src_vocab, second.main.tgt_vocab, second.main.tgt_vocab_rev = second.data_utils.gen_vocab(second.main.WORD_VOCAB_FILE, second.main.LABEL_VOCAB_FILE)
            _, second.main.ulist = second.graph.gen_not_seen(str(PurePath(data_path) / 'complex/'), str(PurePath(data_path) / 'complex/test/'))
            print("Begin Inference:")
            second.main.infer(second.args.params, data_sub_path[model], str(PurePath(model_path[model]) / 'ubest/model.pt'))
            SecondStage.get_result(data_path, model_path[model], variant=model[model.find("_"):-1])

    @staticmethod
    def gen_result_file(data_path: str, model: str):
        def read_data_file(path):
            # read all the data files in the path, one file one item
            data_list = []
            for item in sorted_nt(os.listdir(path)):
                name = str(PurePath(path) / item)
                if(item[0] != '.' and os.path.isfile(name)): 
                    with open(name, 'r') as file_r:
                        data_f = file_r.read().strip()
                    data_list.append(data_f)
    
            return data_list

        def read_infer_file(path):
            # read from the inferred file, one line one item
            data_f = []
            if(os.path.isfile(path)): 
                with open(path, 'r') as file_r:
                    data_f = file_r.read().strip().split('\n')
            
            return data_f

        def gen_infer_file(r_path, w_path, t_list):
            # generate the inferred file
            write_str_list = []
            data_list = read_infer_file(r_path)

            for tokens in t_list:
                str_list = []
                for t in tokens:
                    line = data_list.pop(0)
                    pred = line.strip()
                    str_list.append(t + "\t" + pred)

                write_str_list.append('\n'.join(str_list))
            
            assert len(data_list) == 0

            with open(w_path, mode='w', encoding='utf-8') as write_f:
                write_f.write('\r\n'.join(write_str_list))
                write_f.flush()

def train(stage, data_path):
    if stage == 'first':
        if not os.path.exists('./models/first/out/'):
            os.makedirs('./models/first/out/')
        FirstStage.target(data_path, './models/first/out/')
        FirstStage.vocab(0, data_path, './models/first/out/')
        FirstStage.vocab(1, data_path, './models/first/out/')
        FirstStage.train(data_path)
    elif stage == 'second':
        if not os.path.exists('./models/second/out/'):
            os.makedirs('./models/second/out/')
        SecondStage.target(data_path, './models/second/out/')
        SecondStage.vocab(0, data_path, './models/second/out/')
        SecondStage.vocab(1, data_path, './models/second/out/')
        SecondStage.train(data_path)
    else:
        assert False, "Stage not support now."

def eval(RQ, data_path, model):
    models = model.split(',')
    if RQ == 'RQ1':
        model_path = {
            'TRAINED': './models/first/',
            'STIR': './pretrained/first/stir/',
            'STIR_A': './pretrained/first/stir_a/',
            'DeepTyper': './pretrained/first/deeptyper/'
        }
        for model_ in models:
            print("Preparing model: " + model_)
            FirstStage.target(data_path, str(PurePath(model_path[model_]) / 'out'))
            FirstStage.vocab(0, data_path, str(PurePath(model_path[model_]) / 'out'))
            FirstStage.vocab(1, data_path, str(PurePath(model_path[model_]) / 'out'))
        FirstStage.eval(data_path, models)
        for model_ in models:
            FirstStage.gen_data_for_second(data_path, model_)
    elif RQ == 'RQ2,RQ3':
        if set(models).intersection(set(['TRAINED', 'TRAINED_OT', 'TRAINED_DT', 'TRAINED_GT'])):
            SecondStage.target(data_path, './models/second/out/')
            SecondStage.vocab(0, data_path, './models/second/out/')
            SecondStage.vocab(1, data_path, './models/second/out/')
        if set(models).intersection(set(['STIR', 'STIR_OT', 'STIR_DT', 'STIR_GT'])):
            SecondStage.target(data_path, './pretrained/second/stir/out/')
            SecondStage.vocab(0, data_path, './pretrained/second/stir/out/')
            SecondStage.vocab(1, data_path, './pretrained/second/stir/out/')
        SecondStage.eval(data_path, models, True)
    else:
        assert False, "RQ not support now."

cli_args = argparse.Namespace()

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="STIR: Statistical Type Inference for Incomplete Programs",
        add_help=True,
        allow_abbrev=False
    )
    subparsers = parser.add_subparsers(help="operation to execute", dest="operation")
    parser_train = subparsers.add_parser("train", help="train the model")
    parser_train_stage = parser_train.add_subparsers(help="stage of STIR", dest="stage")
    parser_train_stage1 = parser_train_stage.add_parser("first", help="stage 1: predicting type tags")
    # parser_train_stage1.add_argument("--data", type=str, default=str(PurePath("first/data")), help="path to the data folder")
    parser_train_stage1.add_argument("--data", type=str, default=str(PurePath("data")), help="path to the data folder")
    parser_train_stage2 = parser_train_stage.add_parser("second", help="stage 2: inferring complex types")
    # parser_train_stage2.add_argument("--data", type=str, default=str(PurePath("second/data")), help="path to the data folder")
    parser_train_stage2.add_argument("--data", type=str, default=str(PurePath("data")), help="path to the data folder")

    parser_eval = subparsers.add_parser("eval", help="evaluate the model")
    parser_eval_parsers = parser_eval.add_subparsers(help="research questions", dest="rq")
    parser_eval_rq1 = parser_eval_parsers.add_parser("RQ1", help="RQ1: Predicting Type Tags")
    parser_eval_rq1.add_argument("--data", type=str, default=str(PurePath("data")), help="path to the data folder")
    parser_eval_rq1.add_argument("--model", type=str, default="STIR,STIR_A,DeepTyper", help="name of the model")
    parser_eval_rq3 = parser_eval_parsers.add_parser("RQ2,RQ3", help="RQ2: Inferring Complex Types; RQ3: Inferring Zero-Shot Types")
    parser_eval_rq3.add_argument("--data", type=str, default=str(PurePath("data")), help="path to the data folder")
    parser_eval_rq3.add_argument("--model", type=str, default="STIR,STIR_OT,STIR_DT,STIR_GT", help="name of the model")

    # parser_transfer = subparsers.add_parser("transfer", help="manually transfer the result of first stage")
    # parser_transfer.add_argument("--data", type=str, default=str(PurePath("data")), help="path to the data folder")
    # parser_transfer.add_argument("--model", type=str, default="TRAINED", choices=['TRAINED', 'STIR', 'STIR_A', 'DeepTyper'], help="name of the model")

    cli_args = parser.parse_args()

    if cli_args.operation == "train":
        train(cli_args.stage, cli_args.data)
    elif cli_args.operation == "eval":
        eval(cli_args.rq, cli_args.data, cli_args.model)
    # elif cli_args.operation == "transfer":
    #     FirstStage.gen_data_for_second(cli_args.data, cli_args.model)

if __name__ == "__main__" or __name__ == "main":
    main()