import os
from pathlib import PurePath
import numpy as np

total_grammar = 47#45 47
type1_grammar = 14#14 14
type2_grammar = 33#31 33

st=[0 for _ in range(20)]
Top=1
dic={}
lst=[]
Num=0
flag = 0

Grammar_dict={'S':0, 'N':1, 'E':2, 'Z':3, 'Y':4, 'X':5, 'W':6, 'U':7, 'T':8, 'FUNC':9, 'PTR':10,
              '<eos>':11, 'array':12, 'struct':13, 'union':14, 'int':15,
              'char':16, 'double':17, 'short':18, 'bool':19, 'float':20, 'long':21,
              'longlong':22, 'longdouble':23, 'void':24, 'enum':25, 'O':26}
labelDict={'O':0,'int':1,'PTR':2,'struct':3,'char':4,'double':5,'array':6,'void':7,'enum':8,'short':9,'union':10,'bool':11,'float':12,'long':13,'longlong':14,'longdouble':15,'FUNC':16}

best_path =[[{} for _ in range(20)] for _ in range(20)]
Grammar=[{} for _ in range(total_grammar)]

def init():
    path = PurePath('./second/pcfg/') / 'input.pcfg'
    f = open(path, "r")
    for i in range(total_grammar):
        line = f.readline().strip().split()
        if not line:
            break
        if i < type1_grammar:
            Grammar[i] = {'Num':i, 'Name':line[0], 'first':line[2], 'second':line[3], 'prob':float(line[4])}
        else:
            Grammar[i] = {'Num':i, 'Name':line[0], 'first':line[2], 'second':None, 'prob':float(line[3])}

def Back(type, l, r, cur):
    global Num
    global Top
    global st
    global flag
    if(Grammar[best_path[l][r][type]['path']['rule']]['second'] == None):
        Num = Num+1
        ss = Grammar[best_path[l][r][type]['path']['rule']]['first']
        if ss == '<eos>':
            Top=Top-1
            return
        if ss == 'error':
            return
        dic[Num] = labelDict[ss]
        if st[Top] != 0:
            lst.append((st[Top],Num))
        #print("{%d:%s} (%d,%d)"%(Num,ss,st[Top],Num))
        if ss == 'FUNC' or ss == 'PTR' or ss == 'array' or ss == 'struct' or ss == 'union' or ss == 'enum' or flag == 0:
            if ss == 'FUNC' or ss == 'PTR' or ss == 'array' or ss == 'struct' or ss == 'union' or ss == 'enum':
                flag = 1
            Top = Top + 1
            st[Top] = Num
        return
    Num = Num+1
    #print("{%d:%s} (%d,%d)"%(Num,Grammar[best_path[l][r][type]['path']['rule']]['first'],cur,Num))
    Back(Grammar_dict[Grammar[best_path[l][r][type]['path']['rule']]['first']], l, best_path[l][r][type]['path']['split'], Num)
    Num = Num+1
    #print("{%d:%s} (%d,%d)"%(Num,Grammar[best_path[l][r][type]['path']['rule']]['second'],cur,Num))
    Back(Grammar_dict[Grammar[best_path[l][r][type]['path']['rule']]['second']], best_path[l][r][type]['path']['split'], r, Num)

def get_tree(s):
    global dic
    global lst
    global st # EDITED
    dic = {}
    lst = []
    st=[0 for _ in range(20)]
    word_list = s.split('\t')
    assert len(word_list) > 0
    global Num
    global Top
    global flag

    for i in range(20): 
        for j in range(20):
            for x in range(total_grammar): 
                best_path[i][j][x] = {'prob': 0.0, 'path': {'split': None, 'rule': None}}

    for j in range(len(word_list)):
        for x in range(type1_grammar, total_grammar):
            if Grammar[x]['first'] == word_list[j]:
                best_path[j][j+1][Grammar_dict[Grammar[x]['Name']]] = {'prob':Grammar[x]['prob'], 'path':{'split':None, 'rule':x}}
        for i in range(j-1, -1, -1):
            for k in range(i+1, j+1):
                for x in range(type1_grammar):
                    if(best_path[i][j+1][Grammar_dict[Grammar[x]['Name']]]['prob'] < Grammar[x]['prob'] * best_path[i][k][Grammar_dict[Grammar[x]['first']]]['prob'] * best_path[k][j+1][Grammar_dict[Grammar[x]['second']]]['prob']):
                        best_path[i][j+1][Grammar_dict[Grammar[x]['Name']]] = {'prob':Grammar[x]['prob'] * best_path[i][k][Grammar_dict[Grammar[x]['first']]]['prob'] * best_path[k][j+1][Grammar_dict[Grammar[x]['second']]]['prob'], 'path':{'split':k,'rule':x}}
    Back(Grammar_dict['S'], 0, len(word_list), Num)
    Num = 0
    flag = 0
    Top = 1
    
    return dic, lst