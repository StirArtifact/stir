# coding=utf-8
# constants of attention mode
RNN_ATTENTION_LUONG = 'luong'
RNN_ATTENTION_SLUONG = 'scaled_luong'
RNN_ATTENTION_BAHDANAU = 'bahdanau'
RNN_ATTENTION_NBAHDANAU = 'normed_bahdanau'
# constants of unit type
RNN_UNIT_TYPE_LSTM = 'lstm'
RNN_UNIT_TYPE_GRU = 'gru'
RNN_UNIT_TYPE_LAYER_NORM_LSTM = 'layer_norm_lstm'
########data relate############
MAIN_PATH = 'ner_data/'
WORD_VOC = 'vocab.na.data'
NER_LABEL = 'vocab.lf.data'
# ##################################
# ==== rnn releated ========
# ##################################
class Args():
    def __init__(self):
        self.learning_rate = 0.002
        self.num_layers = 2
        self.num_units = 256
        self.residual = True
        self.batch_first = True
        self.src_vocab_size = 45783
        self.tgt_vocab_size = 21
        self.embedding_size = 128
        self.mlp_hidden = 128
        self.unit_type = RNN_UNIT_TYPE_LSTM
        self.need_atten = False
        self.attention_mode = RNN_ATTENTION_BAHDANAU
        self.dropout = 0.5
        self.forget_bias = 1.0
        self.share_vocab = False
        self.out_dir = './models/'
        self.train_steps = 400000 #train model total steps
        self.train_save_steps = 1000 #saving model steps
        self.opttype = 'Adam'#'SGD'
        self.l2_rate = 0.0001 #use l2 loss
        self.train_type = 'main'
        self.batch_size =  64
        self.decay_step = 100
        self.decay_rate = 0.996#0.996
        self.saver_max_time = 10
        self.read_data_len =1000	#read data len from data file per time
        self.add_noise = False
        self.learning_rate_warmup_steps = 16000
        self.epochs=100

params = Args()