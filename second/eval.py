import time

import argparse
import torch
import numpy as np

import args
import data_utils as du
from main import infer, UNSEEN_FILE
import graph as gh
from PCFG.transfer import recover_data

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="stir", help="data source")
    parser.add_argument("--time", default="false", help="show time")

    arguments = parser.parse_args()
    du.has_simple = True
    is_func = False
    
    trgt_path = './test.lf.data_'
    gh.trgt_path_t = './PCFG/test_lf_data_input.txt'
    print("\n#######################")
    if(arguments.source == "stir"):
        path = 'infer/test/'
        print("Stir:")
    elif(arguments.source == "deeptyper"):
        path = 'deeptAten/test/'
        print("Stir-DT:")
    elif(arguments.source == "origin"):
        path = 'simple/test/'
        print("Stir-GT:")
    elif(arguments.source == "null"):
        path = 'simple/test/'
        du.has_simple = False
        print("Stir-OT:")
    # elif(arguments.source == "func"):
        # path = 'infer/func/'
        # gh.first_trgt = './ner_data/simple/func/'
        # gh.first_pred = './ner_data/infer/func/'
        # UNSEEN_FILE = './ner_data/unseen_func.data'
        # trgt_path = './test.lf.data_func'
        # gh.trgt_path_t = './PCFG/test_lf_data_func_input.txt'
        # is_func = True
        # print("Stir-PF:")
    else:
        assert False, "Source not support now."
    
    print("#######################")
    
    # eval
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    print("Begin Inference:")
    ist = time.time()
    infer(args.params, path, './models_newest/ubest/model.pt', is_func)
    ied = time.time()

    print("\nProcesssing Result...")
    # transfer data
    # if(arguments.source == "func"):
        # rpt = "./test.lf.data_func"
    # else:
    rpt = "./test.lf.data_"
    rpp = "./test.infer.ner"
    wp = "./test_infer_ner.txt"
    rst = time.time()
    recover_data(rpt, rpp, wp)
    red = time.time()

    # process
    pred_path = './test.infer.ner'
    gh.pred_path_t = './test_infer_ner.txt'
    # gh.uset, gh.ulist = gh.gen_not_seen(gh.train_dir, gh.test_dir)
    gh.ulist = du.read_unseen_file(UNSEEN_FILE)
    gh.gen_first_related(gh.first_trgt, gh.first_pred)
    gh.toTree(trgt_path, pred_path)
    
    # print time
    if(arguments.time == "true"):
        ttime = (ied - ist) + (red - rst)
        # if(arguments.source == "func"):
            # cnt = 592
            # print("\nTime takes for each function: ", ttime / cnt)
        # else:
        cnt = 842
        print("\nTime takes for each file: ", ttime / cnt)