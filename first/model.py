# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from tqdm import tqdm
from torchcrf import CRF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    '''Function to initialize weights for models
    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if(m.bias is not None):
            init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell) or isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class Model(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, num_units, mlp_hidden, units_type, dropout, batch_first=True, need_atten=False):
        super(Model, self).__init__()

        self.batch_first = batch_first
        self.need_atten = need_atten

        self.embedding = nn.Embedding(src_vocab_size, embedding_size)

        if(units_type == 'lstm'):
            self.encoder = nn.LSTM(embedding_size, num_units, 1, bidirectional=True)
            self.decoder = nn.LSTM(embedding_size + 2*num_units, num_units, 2, dropout=dropout)
        elif(units_type == 'gru'):
            self.encoder = nn.GRU(embedding_size, num_units, 1, bidirectional=True)
            self.decoder = nn.GRU(embedding_size + 2*num_units, num_units, 2, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

        if(need_atten):
            self.atten = nn.Linear(2*num_units, embedding_size)

        self.output = nn.Linear(num_units, tgt_vocab_size)
        self.crf = CRF(tgt_vocab_size)

        self.init_weight()
    
    def init_weight(self):
        self.embedding.apply(weights_init)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.output.apply(weights_init)
    
    def proc_inp(self, inp):
        enc_inp = inp
        dec_inp = inp
        for idx, ip in enumerate(dec_inp):
            try:
                ip.remove(2)
            except:
                print(ip)
            ip.insert(0, 1)

        return enc_inp, dec_inp
    
    def seq2seq(self, enc_out, dec_inp):
        enc_out = enc_out.transpose(0, 1) # batch, seq, hidden
        dec_inp = dec_inp.transpose(0, 1) # batch, seq, hidden
        enc_atten = self.atten(enc_out) # batch, seq, hidden

        batch, n_step, inp_hidden = dec_inp.size()
        
        dec_inp_exp = dec_inp.unsqueeze(2).expand(batch, n_step, n_step, inp_hidden) # batch, seq, seq, hidden
        enc_atten_exp = enc_atten.unsqueeze(1).expand(batch, n_step, n_step, inp_hidden) # batch, seq, seq, hidden

        atten_score = torch.mul(dec_inp_exp, enc_atten_exp).sum(-1) # batch, seq, seq
        atten_weight = F.softmax(atten_score, dim=2) # batch, seq, seq
        context = torch.bmm(atten_weight, enc_out) # batch, seq, hidden

        dec_out, _ = self.decoder(torch.cat([dec_inp, context], dim=2).transpose(0, 1))# seq, batch, hidden

        return dec_out
    
    def forward(self, inp, tags=None):
        enc_inp, dec_inp = self.proc_inp(inp)

        # enc_inp = self.embedding(torch.LongTensor(enc_inp).cuda())
        # dec_inp = self.embedding(torch.LongTensor(dec_inp).cuda())
        # if(tags is not None):
        #     tags = torch.LongTensor(tags).cuda()

        enc_inp = self.embedding(torch.LongTensor(enc_inp).to(device))
        dec_inp = self.embedding(torch.LongTensor(dec_inp).to(device))
        if(tags is not None):
            tags = torch.LongTensor(tags).to(device)

        if(self.batch_first):
            enc_inp = enc_inp.transpose(0, 1) # seq_len, batch, hidden
            dec_inp = dec_inp.transpose(0, 1)
            if(tags is not None):
                tags = tags.transpose(0, 1)
        
        enc_out, _ = self.encoder(enc_inp)
        enc_out = self.dropout(enc_out)
        if(not self.need_atten):
            dec_out, _ = self.decoder(torch.cat([dec_inp, enc_out], dim=2))
        else:
            # TODO need fix
            dec_out = self.seq2seq(enc_out, dec_inp)
        
        emissions = self.output(dec_out) # seq, batch, class

        if(self.training):
            assert tags != None
            if(isinstance(self.encoder, nn.LSTM)):
                return -self.crf(emissions, tags)
            else:
                return F.cross_entropy(emissions.transpose(1, 2), tags)
        else:
            if(isinstance(self.encoder, nn.LSTM)):
                return self.crf.decode(emissions)
            else:
                return torch.argmax(emissions.transpose(0, 1), dim=-1).tolist()