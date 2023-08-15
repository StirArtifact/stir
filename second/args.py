# coding=utf-8

########data relate############
MAIN_PATH = 'ner_data/'
WORD_VOC = 'vocab.na.data'
NER_LABEL = 'vocab.lf.data_'
# ##################################
# ==== rnn releated ========
# ##################################
class Args():
    def __init__(self):
        self.learning_rate = 0.002
        self.num_layers = 3
        self.num_tags = 10
        self.num_units = 256
        self.num_heads = 2
        self.batch_first = True
        self.src_vocab_size = 45781
        self.tgt_vocab_size = 19
        self.embedding_size = 200
        self.mlp_hidden = 128
        self.dropout = 0.5
        self.share_vocab = False
        self.out_dir = './models/'
        self.opttype = 'Adam'#'SGD'
        self.l2_rate = 0.0001 #use l2 loss
        self.batch_size = 16
        self.decay_step = 100
        self.decay_rate = 0.996#0.996
        self.read_data_len =1000	#read data len from data file per time
        self.epochs=1000

params = Args()