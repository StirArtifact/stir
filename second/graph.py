import os
import re
import sys
import subprocess
from copy import deepcopy

from pathlib import PurePath
from rich import print as printr
from tqdm import tqdm

from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph

# import second.PCFG as PCFG
import second.pcfg.pcfg_repair_mp as PCFG
from second.data_utils import sorted_nt 

jobs = 1

labelDict = {'O':0, 'int':1, '*':2, 'struct':3, 'char':4, \
'double':5, 'array':6, 'void':7, 'enum':8, 'short':9, \
'union':10, 'bool':11, 'float':12,'long':13, 'long long':14, \
'long double':15, '->':16}

first_trgt = os.path.join('./ner_data/', 'simple/test/')
first_pred = os.path.join('./ner_data/', 'infer/test/')
# trgt_path = './lf.recover.data'
trgt_path = './test.lf.data_'
trgt_path_t = './PCFG/test_lf_data_input.txt'
# pred_path = './models_newest/ubest/infer.recover.data_trgt'
# pred_path = './infer.recover.data'
pred_path = './models_newest/ubest/test.infer.ner_O'
pred_path_t = './PCFG/test_infer_ner_O_input.txt'
# pred_path_t = './PCFG/best.txt'
train_dir = os.path.join('./ner_data/', 'complex/')
test_dir = os.path.join('./ner_data/', 'complex/test/')

uset = set()
ulist = list()
prim_list = list()
ptr_list = list()
stru_list = list()
func_list = list()
fw_list = list()

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
        param = label[left+1:idx]
        ret = label[idx+2:right]
        orig = label[left:right+1]
        new = '->' + ret + ',' + param
        newstr = newstr.replace(orig, new, 1)
    
    return newstr

def proc_enum(label):
    if label.find('enum') >= 0:
        label = re.sub(r'enum\([int\,]+\)', 'enum', label)
    
    return label

def proc_label(label):
    label = label.strip()
    label = label.replace(',)', ')')
    label = label.replace('struct()', 'struct')
    label = label.replace('union()', 'union')

    return label

class Node:
    def __init__(self, token):
        self.token = token

class LabelGraph:
    def __init__(self, label):
        self.labelV = []
        self.E = []

        # label = label.strip()
        # label = label.replace(',)', ')')
        # label = label.replace('struct()', 'struct')
        # label = label.replace('union()', 'union')
        
        # print(label)
        root = Node('mock')
        self.labelV.append(root)
        _ = self._parse_label(self, root, label, 0)

        # edge list
        self.edgeList = []

    @staticmethod
    def _parse_label(self, root, label, idx):
        l_type = ''
        # preN = root

        i = idx
        while i < len(label):
            if(label[i] == ','):
                if(l_type != '' and l_type in labelDict):
                    preN = Node(l_type)
                    self.labelV.append(preN)
                    self.E.append([root, preN])
                l_type = ''
            elif(label[i] == '('):
                if(l_type != '' and l_type in labelDict):
                    preN = Node(l_type)
                    self.labelV.append(preN)
                    self.E.append([root, preN])
                i = self._parse_label(self, preN, label, i+1)
                l_type = ''
            elif(label[i] == ')'):
                if(l_type != '' and l_type in labelDict):
                    preN = Node(l_type)
                    self.labelV.append(preN)
                    self.E.append([root, preN])
                l_type = ''
                return i
            else:
                l_type += label[i]
            i += 1
        
        if(l_type != '' and l_type in labelDict):
            l_n = Node(l_type)
            self.labelV.append(l_n)
            self.E.append([root, l_n])
        
        return len(label)
    
    def _get_graph(self):
        self.nodeDict = {}
        # self.nodeDict[0] = -1
        for idx, labelV in enumerate(self.labelV[1:]):
            self.nodeDict[idx] = labelDict[labelV.token]
        
        for edge in self.E[1:]:
            self.edgeList.append((self.labelV.index(edge[0])-1, self.labelV.index(edge[1])-1))


def toGraph(trgt_p, pred_p):
    total_sim = 0.0
    unseen_sim = 0.0
    prim_sim = 0.0
    ptr_sim = 0.0
    stru_sim = 0.0
    func_sim = 0.0
    # fw_sim = 0.0
    uprim_sim = 0.0
    uptr_sim = 0.0
    ustru_sim = 0.0
    ufunc_sim = 0.0
    # ufw_sim = 0.0
    num = 0
    unseen_num = 0
    ptr_num = 0
    stru_num = 0
    func_num = 0
    uptr_num = 0
    ustru_num = 0
    ufunc_num = 0

    lines1 = open(trgt_p, 'r', encoding='utf-8').readlines()
    lines2 = open(pred_p, 'r', encoding='utf-8').readlines()
    file_w = open('./type_sim.data', 'w', encoding='utf-8')

    assert len(lines1) == len(lines2)
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        l1 = l1.strip()
        l2 = l2.strip()   
        l1 = proc_label(l1)
        l2 = proc_label(l2)
        if l1 == 'O' or i in prim_list:
            continue
        
        if l1.find('(') == -1:
            sim = 1.0 if l1 == l2 else 0.0
            # continue
        elif l1.find('(') != -1 and l2.find('(') == -1:
            sim = 0.5 if l1.startswith(l2) else 0.0
        else:
            lg1 = LabelGraph(l1)
            lg1._get_graph()
            lg2 = LabelGraph(l2)
            lg2._get_graph()

            lg1_nodes = lg1.nodeDict
            lg1_edges = lg1.edgeList
            lg2_nodes = lg2.nodeDict
            lg2_edges = lg2.edgeList
            
            G1 = Graph(lg1_edges, node_labels=lg1_nodes)
            G2 = Graph(lg2_edges, node_labels=lg2_nodes)
            wl_kernel = WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=True)
            wl_kernel.fit_transform([G1])
            sim = wl_kernel.transform([G2]).item()

        total_sim += sim
        num += 1
        if i in ptr_list:
            ptr_sim += sim
            ptr_num += 1
        elif i in stru_list:
            stru_sim += sim
            stru_num += 1
        elif i in func_list:
            func_sim += sim
            func_num += 1
        else:
            prim_sim += sim
        
        # if i in fw_list:
        #     fw_sim += sim
        
        if i in ulist and i not in prim_list:
            unseen_sim += sim
            unseen_num += 1
            if i in ptr_list:
                uptr_sim += sim
                uptr_num += 1
            elif i in stru_list:
                ustru_sim += sim
                ustru_num += 1
            elif i in func_list:
                ufunc_sim += sim
                ufunc_num += 1
            else:
                uprim_sim += sim
            
            # if i in fw_list:
            #     ufw_sim += sim
        
            file_w.write('idx ' + str(i) + ': \n' + l1 + '\n' + l2 + '\n\n')
    
    print('\naverage sim: ', total_sim / num)
    # print('average primitive sim: ', prim_sim / len(prim_list))
    print('average pointer sim: ', ptr_sim / ptr_num)
    print('average structure sim: ', stru_sim / stru_num)
    print('average function sim: ', func_sim / func_num)
    # print('average first wrong sim: ', fw_sim / len(fw_list))

    print('\naverage zero-shot sim: ', unseen_sim / unseen_num)
    # print('average zero-shot primitive sim: ', uprim_sim / len(set(prim_list) & set(ulist)))
    print('average zero-shot pointer sim: ', uptr_sim / len(set(ptr_list) & set(ulist)))
    print('average zero-shot structure sim: ', ustru_sim / len(set(stru_list) & set(ulist)))
    print('average zero-shot function sim: ', ufunc_sim / len(set(func_list) & set(ulist)))
    # print('average zero-shot first wrong sim: ', ufw_sim / len(set(fw_list) & set(ulist)))

def toTree(trgt_p, pred_p, check_ot=False, underline=False):
    total_sim = 0.0
    unseen_sim = 0.0
    prim_sim = 0.0
    ptr_sim = 0.0
    stru_sim = 0.0
    func_sim = 0.0
    fw_sim = 0.0
    uprim_sim = 0.0
    uptr_sim = 0.0
    ustru_sim = 0.0
    ufunc_sim = 0.0
    ufw_sim = 0.0
    num = 0
    unseen_num = 0
    ptr_num = 0
    stru_num = 0
    func_num = 0
    uptr_num = 0
    ustru_num = 0
    ufunc_num = 0

    lines1 = open(trgt_p, 'r', encoding='utf-8').readlines()
    lines2 = open(pred_p, 'r', encoding='utf-8').readlines()
    # lines3 = open(trgt_path_t, 'r', encoding='utf-8').readlines()
    # lines4 = open(pred_path_t, 'r', encoding='utf-8').readlines()
    PCFG.set_num_threads(jobs)

    # PCFG.init()
    with open("second/pcfg/grammar.erronly.O.RMSE0.00905285.txt", 'r', encoding='utf-8') as f:
        grammar = f.read()
    parser = PCFG.Parser(grammar)

    with open("second/pcfg/grammar.erreos.O.RMSE0.00491918.txt", "r", encoding="utf-8") as f:
        grammar_loose = f.read()
    parser_loose = PCFG.Parser(grammar_loose)

    # print("PCFG parser initialized.")
    assert len(lines1) == len(lines2)
    # assert len(lines3) == len(lines4)
    for i, (l1, l2) in tqdm(enumerate(zip(lines1, lines2)), total=len(lines1), ncols=80):
    # for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        l1 = l1.strip()
        l2 = l2.strip()   
        if l1 == 'O' or i in prim_list:
            continue
        
        if l1.find('->') >= 0:
            l1 = l1.replace('->', 'func')
        if l1.find('*') >= 0:
            l1 = l1.replace('*', 'ptr')
        if l1.find('O') >= 0:
            l1 = l1.replace('O', 'o')
        if l2.find('->') >= 0:
            l2 = l2.replace('->', 'func')
        if l2.find('*') >= 0:
            l2 = l2.replace('*', 'ptr')
        if l2.find('O') >= 0:
            l2 = l2.replace('O', 'o')

        splits1 = l1.split('\t')
        
        if len(splits1) == 1:
            sim = 1.0 if l1 == l2 else 0.0
        else:
            if len(splits1) > 10:
                splits1 = splits1[:10]
            data1 = '\t'.join(splits1)
            data2 = l2

            # print("line %d " % i, end="")
            # print(data1, data2)
            lg1_nodes, lg1_edges = parser.get_tree(data1, ignore_err_nodes=False, err_to_tree=True)
            lg2_nodes, lg2_edges = parser.get_tree(data2, ignore_err_nodes=False, err_to_tree=True)
            if len(lg1_edges) == 0 or len(lg2_edges) == 0:
                # print("try strict grammar without error nodes failed")
                lg1_nodes, lg1_edges = parser.get_tree(data1, ignore_err_nodes=False, err_to_tree=False)
                lg2_nodes, lg2_edges = parser.get_tree(data2, ignore_err_nodes=False, err_to_tree=False) # try strict grammar with error nodes

            assert len(lg1_edges) != 0 and len(lg2_edges) != 0, (l1, l2)
            
            G1 = Graph(lg1_edges, node_labels=lg1_nodes)
            G2 = Graph(lg2_edges, node_labels=lg2_nodes)
            wl_kernel = WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=True)
            wl_kernel.fit_transform([G1])
            sim = wl_kernel.transform([G2]).item()
            # print(i, sim)

        if not check_ot:
            if i in ulist and i not in prim_list:
                unseen_sim += sim
                unseen_num += 1
                if i in ptr_list:
                    uptr_sim += sim
                    uptr_num += 1
                elif i in stru_list:
                    ustru_sim += sim
                    ustru_num += 1
                elif i in func_list:
                    ufunc_sim += sim
                    ufunc_num += 1
                else:
                    uprim_sim += sim

                # if i in fw_list:
                #     ufw_sim += sim
        total_sim += sim
        num += 1
        if i in ptr_list:
            ptr_sim += sim
            ptr_num += 1
        elif i in stru_list:
            stru_sim += sim
            stru_num += 1
        elif i in func_list:
            func_sim += sim
            func_num += 1
        else:
            prim_sim += sim

    # assert len(lines3) == len(lines4) == 0

    print("Graph similarity for complex types: ")
    if underline:
        printr('Pointer: [bold underline]{:.2%}[/bold underline]'.format(ptr_sim / ptr_num))
        printr('Structure: [bold underline]{:.2%}[/bold underline]'.format(stru_sim / stru_num))
        printr('Function: [bold underline]{:.2%}[/bold underline]'.format(func_sim / func_num))
        printr('Macro Avg: [bold underline]{:.2%}[/bold underline]'.format(total_sim / num))
    else:
        printr('Pointer: [bold]{:.2%}[/bold]'.format(ptr_sim / ptr_num))
        printr('Structure: [bold]{:.2%}[/bold]'.format(stru_sim / stru_num))
        printr('Function: [bold]{:.2%}[/bold]'.format(func_sim / func_num))
        printr('Macro Avg: [bold]{:.2%}[/bold]'.format(total_sim / num))

    if not check_ot:
        print("\nGraph similarity for zero-shot types: ")
        if underline:
            printr('Pointer: [bold underline]{:.2%}[/bold underline]'.format(uptr_sim / uptr_num))
            printr('Structure: [bold underline]{:.2%}[/bold underline]'.format(ustru_sim / ustru_num))
            printr('Function: [bold underline]{:.2%}[/bold underline]'.format(ufunc_sim / ufunc_num))
            printr('Macro Avg: [bold underline]{:.2%}[/bold underline]'.format(unseen_sim / unseen_num))
        else:
            printr('Pointer: [bold]{:.2%}[/bold]'.format(uptr_sim / uptr_num))
            printr('Structure: [bold]{:.2%}[/bold]'.format(ustru_sim / ustru_num))
            printr('Function: [bold]{:.2%}[/bold]'.format(ufunc_sim / ufunc_num))
            printr('Macro Avg: [bold]{:.2%}[/bold]'.format(unseen_sim / unseen_num))

def type_prefix_score(target_list, predicated_list, average=False):
    smaller_len = min(len(target_list), len(predicated_list))
    for i in range(smaller_len):
        if target_list[i] != predicated_list[i]:
            return (i / len(target_list)) if average else i
    return (smaller_len / len(target_list)) if average else smaller_len


def get_type_prefix_score(trgt_p, pred_p, trgt_pcfg_path, pred_pcfg_path, check_unseen=False):
    total_tps = 0.0
    unseen_tps = 0.0
    ptr_tps = 0.0
    stru_tps = 0.0
    func_tps = 0.0
    uptr_tps = 0.0
    ustru_tps = 0.0
    ufunc_tps = 0.0
    total_atps = 0.0
    unseen_atps = 0.0
    ptr_atps = 0.0
    stru_atps = 0.0
    func_atps = 0.0
    uptr_atps = 0.0
    ustru_atps = 0.0
    ufunc_atps = 0.0
    num = 0
    unseen_num = 0
    ptr_num = 0
    stru_num = 0
    func_num = 0
    uptr_num = 0
    ustru_num = 0
    ufunc_num = 0

    lines1 = open(trgt_p, 'r', encoding='utf-8').readlines()
    lines2 = open(pred_p, 'r', encoding='utf-8').readlines()
    lines3 = open(trgt_pcfg_path, 'r', encoding='utf-8').readlines()
    lines4 = open(pred_pcfg_path, 'r', encoding='utf-8').readlines()

    assert len(lines1) == len(lines2)
    assert len(lines3) == len(lines4)
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        l1 = l1.strip()
        l2 = l2.strip()
        if l1 == 'O' or i in prim_list:
            continue

        splits1 = l1.split('\t')

        if len(splits1) == 1:
            continue
        else:
            t_str = lines3.pop(0).strip().split('\t')[:10]
            p_str = lines4.pop(0).strip().split('\t')[:10]

            tps = type_prefix_score(t_str, p_str)
            avg_tps = type_prefix_score(t_str, p_str, average=True)
            # if check_unseen:
            if i in ulist and i not in prim_list:
                unseen_tps += tps
                unseen_atps += avg_tps
                unseen_num += 1
                if i in ptr_list:
                    uptr_tps += tps
                    uptr_atps += avg_tps
                    uptr_num += 1
                elif i in stru_list:
                    ustru_tps += tps
                    ustru_atps += avg_tps
                    ustru_num += 1
                elif i in func_list:
                    ufunc_tps += tps
                    ufunc_atps += avg_tps
                    ufunc_num += 1
            # else:
            total_tps += tps
            total_atps += avg_tps
            num += 1
            if i in ptr_list:
                ptr_tps += tps
                ptr_atps += avg_tps
                ptr_num += 1
            elif i in stru_list:
                stru_tps += tps
                stru_atps += avg_tps
                stru_num += 1
            elif i in func_list:
                func_tps += tps
                func_atps += avg_tps
                func_num += 1
    assert len(lines3) == len(lines4) == 0

    # if check_unseen:
    print('\nAverage pointer TPS: ', ptr_tps / ptr_num)
    print('Average structure tps: ', stru_tps / stru_num)
    print('Average function tps: ', func_tps / func_num)
    print('Average TPS: ', total_tps / num)

    print('\nAverage pointer TPS Avg: ', ptr_atps / ptr_num)
    print('Average structure tps Avg: ', stru_atps / stru_num)
    print('Average function tps Avg: ', func_atps / func_num)
    print('Average TPS Avg: ', total_atps / num)

    print('\nAverage zero-shot pointer TPS: ', uptr_tps / uptr_num)
    print('Average zero-shot structure tps: ', ustru_tps / ustru_num)
    print('Average zero-shot function tps: ', ufunc_tps / ufunc_num)
    print('Average zero-shot TPS: ', unseen_tps / unseen_num)

    print('\nAverage zero-shot pointer TPS Avg: ', uptr_atps / ustru_num)
    print('Average zero-shot structure tps Avg: ', ustru_atps / ustru_num)
    print('Average zero-shot function tps Avg: ', ufunc_atps / ufunc_num)
    print('Average zero-shot TPS Avg: ', unseen_atps / unseen_num)
    # else:

def read_dir(d):
    types = set()
    labels = dict()
    count = 0
    for item in sorted_nt(os.listdir(d)):
        name = str(PurePath(d) / item)
        if(item[0] != '.' and os.path.isfile(name)): 
            f = open(name, 'r')
            lines = f.read().strip().split('\n')
            if(len(lines) > 1000):
                continue
            for line in lines:
                splits = line.strip().rsplit('\t', 1)
                if(len(splits) <= 1):
                    continue
                type_ = splits[1]
                types.add(type_)
                labels[count] = type_
                count += 1
    
    return types, labels

def gen_not_seen(t_dir, i_dir):
    train_set, _ = read_dir(t_dir)
    test_set, test_dict = read_dir(i_dir)

    unseen_set = test_set - train_set
    unseen_list = [k for k, v in test_dict.items() if v in unseen_set]
    assert len(test_dict) == 253108

    return unseen_set, unseen_list

def read_first(d):
    first_list = []
    items = sorted_nt(os.listdir(d))
    for item in items:
        file = str(PurePath(d) / item)
        lines = []
        line_num = 0
        for line in open(file, 'r', encoding='utf-8'):
            splits = line.strip().rsplit('\t', 1)
            lines.append(splits[1])
            line_num += 1
        
        if(line_num <= 1000):
            first_list.extend(lines)
    
    return first_list

def gen_first_related(t_dir, i_dir):
    trgt_first = read_first(t_dir)
    pred_first = read_first(i_dir)
    assert len(trgt_first) == len(pred_first)

    for i, (trgt, pred) in enumerate(zip(trgt_first, pred_first)):
        if trgt == 'O':
            continue
        if i < 3:
            continue
        
        if trgt == 'pointer' or trgt == 'array':
            ptr_list.append(i)
        elif trgt == 'struct' or trgt == 'union' or trgt == 'enum':
            stru_list.append(i)
        elif trgt == 'function':
            func_list.append(i)
        else:
            prim_list.append(i)
        
        if trgt != pred:
            fw_list.append(i)


if __name__ == '__main__':
    uset, ulist = gen_not_seen(train_dir, test_dir)
    gen_first_related(first_trgt, first_pred)
    # toGraph(trgt_path, pred_path)
    toTree(trgt_path, pred_path)