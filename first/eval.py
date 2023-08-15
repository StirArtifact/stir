import os
import time

import argparse
import torch
import numpy as np

import first.args as args
import first.main as main
from first.main import infer
from first.data_utils import check_result_multi_class

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stir", help="name of model")
    parser.add_argument("--atten", default="false", help="need attention or not")
    parser.add_argument("--time", default="false", help="show data")

    arguments = parser.parse_args()
    print("\n#######################")

    path = './'
    if(arguments.model == "func"):
        label_path = os.path.join(path, 'test.lf.data_func')
    else:
        label_path = os.path.join(path, 'test.lf.data')
    infer_path = os.path.join(path, 'test.infer.ner')
    is_func = False

    if(arguments.model == "stir"):
        args.params.unit_type = args.RNN_UNIT_TYPE_LSTM
        if(arguments.atten == "false"):
            args.params.need_atten = False
            main.SAVE_PATH = os.path.join(path, 'models_lstm_pre0.9607_rec1.0/best/model.pt')
            print("Stir:")
        elif(arguments.atten == "true"):
            args.params.need_atten = True
            args.params.batch_size = 16
            main.SAVE_PATH = os.path.join(path, 'models_lstm_atten_pre0.8707_rec1.0/best/model.pt')
            print("Stir-A:")
        else:
            assert False, "Atten type not support now."
    elif(arguments.model == "deeptyper"):
        args.params.unit_type = args.RNN_UNIT_TYPE_GRU
        if(arguments.atten == "false"):
            assert False, "Wrong parameter!"
        elif(arguments.atten == "true"):
            args.params.need_atten = True
            args.params.batch_size = 16
            main.SAVE_PATH = os.path.join(path, 'models_deeptyper_consis/best/model.pt')
            print("DeepTyper:")
        else:
            assert False, "Atten type not support now."
    elif(arguments.model == "func"):
        args.params.unit_type = args.RNN_UNIT_TYPE_LSTM
        args.params.need_atten = False
        main.SAVE_PATH = os.path.join(path, 'models_lstm_pre0.9607_rec1.0/best/model.pt')
        is_func = True
        print("Stir-P:")
    else:
        assert False, "Model not support now."

    print("#######################")

    # eval
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    print("Begin Inference:")
    st = time.time()
    infer(args.params, is_func)
    ed = time.time()
    print("\nProcesssing Result...")
    print("\nSimpleType:")
    check_result_multi_class(label_path, infer_path, mode='simple')
    print("\nComplexType:")
    check_result_multi_class(label_path, infer_path, mode='complex')
    print("\nAllType:")
    check_result_multi_class(label_path, infer_path, mode='all')
    
    
    # print time
    if(arguments.time == "true"):
        if(arguments.model == "func"):
            print("\nTime takes for each function: ", (ed - st) / 592)
        else:
            print("\nTime takes for each file: ", (ed - st) / 841)