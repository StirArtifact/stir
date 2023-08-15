import math
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
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
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src

        for mod in self.layers:
            output, attn_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weight

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weight

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ntype, ninp, nhead, nhid, nlayers, ntags, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.ninp = ninp
        self.ntags = ntags

        self.word_encoder = nn.Embedding(ntoken, ninp)
        self.type_encoder = nn.Embedding(ntype, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = CustomEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.lstm_decoder = nn.LSTMCell(2*ninp, ninp)

        self.output = nn.Linear(ninp, ntype)
        self.drop = nn.Dropout(dropout)

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

        self.init_weights()

    def _generate_key_padding_mask(self, t):
        # zero = torch.zeros_like(t).cuda()
        zero = torch.zeros_like(t).to(device)
        mask = torch.where(t != 1, t, zero) == 0
        return mask

    def init_weights(self):
        # initrange = 0.1
        # nn.init.uniform_(self.word_encoder.weight, -initrange, initrange)
        # nn.init.uniform_(self.type_encoder.weight, -initrange, initrange)
        self.output.apply(weights_init)
        self.lstm_decoder.apply(weights_init)
    
    def decode(self, src, case, weight):
        # src and case size: [bsz, seq, em]
        # weight size: [bsz, seq, seq]
        bsz, seq, _ = case.size()
        # list of [bsz, seq, em], lenth = mtag
        out_list = [self.output(case)]
        h_n = torch.zeros(bsz*seq, self.ninp, dtype=case.dtype, device=case.device) # bsz*seq, em
        c_n = torch.zeros(bsz*seq, self.ninp, dtype=case.dtype, device=case.device)
        for i in range(self.ntags-1):
            inp = torch.cat([src, case], dim=2)
            inp = inp.view(-1, 2*self.ninp) # bsz*seq, em
            h_n, c_n = self.lstm_decoder(inp, (h_n, c_n)) # bsz*seq, em
            out = torch.bmm(weight, h_n.view(bsz, seq, self.ninp)) # bsz, seq, em
            out = self.output(self.drop(out))
            out_list.append(out)

            typ = out.argmax(-1)
            case = self.type_encoder(typ) * math.sqrt(self.ninp)
            case = self.pos_encoder(case) # bsz, seq, em
        
        return torch.stack(out_list, dim=2)

    def forward(self, src, case, tag=None):
        # to tensor
        # src = torch.LongTensor(src).cuda()
        # case = torch.LongTensor(case).cuda() # bsz, seq
        # if tag is not None:
        #     tag = torch.LongTensor(tag).cuda() # bsz, seq, mtag

        src = torch.LongTensor(src).to(device)
        case = torch.LongTensor(case).to(device) # bsz, seq
        if tag is not None:
            tag = torch.LongTensor(tag).to(device) # bsz, seq, mtag

        # gen padding mask
        src_mask = self._generate_key_padding_mask(src)

        # embedding
        src = self.word_encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1).contiguous() # seq, bsz, em
        case = self.type_encoder(case) * math.sqrt(self.ninp)
        case = self.pos_encoder(case) # bsz, seq, em

        # Transformer encoder for attention weight score -> bsz, seq, seq
        _, attn_weight = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        # LSTM decoder for output
        src = src.transpose(0, 1).contiguous() # bsze, seq, em
        dec_out = self.decode(src, case, attn_weight)

        # emissions = self.output(dec_out) # seq, batch, mtag, class

        if(self.training):
            assert tag != None
            dec_out = dec_out.permute(0, 3, 1, 2)
            return self.loss(dec_out, tag)
        else:
            return dec_out.argmax(-1).tolist(), attn_weight.squeeze(0).detach().cpu()