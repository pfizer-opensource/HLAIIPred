import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, pep_block_size, **kwargs):
        self.vocab_size = vocab_size
        self.pep_block_size = pep_block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class EncoderBlock(nn.Module):
    """A transformer encoder block in the vanilla transformer"""
    def __init__(self, config):
        super().__init__()
    

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_heads, config.attn_pdrop, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)

    def forward(self, x, x_mask, need_weights=True):
        """
        x: shape(b,lx,h)
        x_mask: shape(b,lx)  1 to be ignored
        """
        # attention part
        attn_output, aw = self.attn(x, x, x, key_padding_mask=x_mask, need_weights=need_weights)  # b, lx, h
        residual = self.ln1(x + attn_output)
        x = self.resid_drop1(residual)

        # feed forward part
        ff_output = self.mlp(x)         # b, lx, h
        residual = self.ln2(x + ff_output)
        x = self.resid_drop2(residual)
        return x, aw

class DecoderBlock(nn.Module):
    """A transformer decoder block in the vanilla transformer"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.self_attn = nn.MultiheadAttention(config.n_embd, config.n_heads, config.attn_pdrop, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.n_embd, config.n_heads, config.attn_pdrop, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)
        self.resid_drop3 = nn.Dropout(config.resid_pdrop)
        # self.register_buffer("causal_mask", ~torch.tril(torch.ones(config.out_block_size-1, config.out_block_size-1)).bool())

    def forward(self, y, y_mask, x_latent, x_mask):
        """
        y: shape(b,ly,h)
        x_latent: shape(b,lx,h)
        x_mask: shape(b,lx)  1 to be ignored
        """
        # self attention part
        # target_len = y.shape[1]
        # self_attn_mask = self.causal_mask[:target_len, :target_len]
        self_attn_output, _ = self.self_attn(y, y, y, key_padding_mask=y_mask, need_weights=False)#, attn_mask=self_attn_mask)
        residual = self.ln1(y + self_attn_output)
        y = self.resid_drop1(residual)

        # cross attention part
        cross_attn_output, att_weights = self.cross_attn(y, x_latent, x_latent, key_padding_mask=x_mask, need_weights=True)
        residual = self.ln2(y + cross_attn_output)
        y = self.resid_drop2(residual)

        # feed forward part
        ff_output = self.mlp(y)
        residual = self.ln3(y + ff_output)
        y = self.resid_drop3(residual)
        
        return y, att_weights

class DataParallelWrappedModel(nn.Module):
    """ a wrapper class to allow data parallel execution of a model"""
    def __init__(self, model, devices) -> None:
        super().__init__()
        self.primal_model = model
        self.parallel_model = nn.DataParallel(model, devices)

    def forward(self, p, p_mask, a, a_mask, k_mask, y=None):
        return self.parallel_model(p, p_mask, a, a_mask, k_mask, y)

    def generate_tokenwise(self, codon, softmax=True, out_mask=None):
        b, t = codon.size()
        x = self.parallel_model(codon)
        x = x[:, -1, :].unsqueeze(1)   # B,1,vocab_size

        if out_mask is not None:
            out_mask = out_mask[:,t-1,:].unsqueeze(1)     # t is the length of the sequence, mask is 0-indexed (index 0 for the first 'codon')
            x = x.masked_fill(out_mask == 0, float('-inf'))

        # softmax
        probs = None
        top_i = None

        if softmax:

            probs = nn.functional.softmax(x, dim=-1)           # B,1,vocab_size

            # sample from the output as a multinomial distribution
            top_i = torch.multinomial(probs.squeeze(1), num_samples=1)      # B, 1
            # one_hot = nn.functional.one_hot(top_i, self.vocab_size)     # B, 1, vocab_size
            # one_hot = one_hot.type(x.dtype)
        
        return x, probs, top_i

    def configure_optimizers(self, train_config):
        return self.primal_model.configure_optimizers(train_config)

class DistributedDataParallelWrappedModel(nn.Module):
    """ a wrapper class to allow data parallel execution of a model"""
    def __init__(self, model, devices) -> None:
        super().__init__()
        self.primal_model = model
        self.parallel_model = nn.DataParallel(model, devices)

    def forward(self, p, p_mask, a, a_mask, k_mask, y=None):
        return self.parallel_model(p, p_mask, a, a_mask, k_mask, y)

    def generate_tokenwise(self, codon, softmax=True, out_mask=None):
        b, t = codon.size()
        x = self.parallel_model(codon)
        x = x[:, -1, :].unsqueeze(1)   # B,1,vocab_size

        if out_mask is not None:
            out_mask = out_mask[:,t-1,:].unsqueeze(1)     # t is the length of the sequence, mask is 0-indexed (index 0 for the first 'codon')
            x = x.masked_fill(out_mask == 0, float('-inf'))

        # softmax
        probs = None
        top_i = None

        if softmax:

            probs = nn.functional.softmax(x, dim=-1)           # B,1,vocab_size

            # sample from the output as a multinomial distribution
            top_i = torch.multinomial(probs.squeeze(1), num_samples=1)      # B, 1
            # one_hot = nn.functional.one_hot(top_i, self.vocab_size)     # B, 1, vocab_size
            # one_hot = one_hot.type(x.dtype)
        
        return x, probs, top_i

    def configure_optimizers(self, train_config):
        return self.primal_model.configure_optimizers(train_config)

class DeConvolutionalAttention(nn.Module):
    """ a transformer model """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # input/output embedding stem
        self.emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.pos_emb_p = nn.Parameter(torch.zeros(1,config.pep_block_size, config.n_embd))  # 1,lp,c
        self.drop_emb_p = nn.Dropout(config.embd_pdrop)
        self.pos_emb_a = nn.Parameter(torch.zeros(1,1,config.allele_block_size, config.n_embd))  # 1,1,la,c
        self.drop_emb_a = nn.Dropout(config.embd_pdrop)

        # encoder blocks
        self.encoder_a = nn.ModuleList([EncoderBlock(config)]*config.n_layer)
        # self.encoder_p = nn.ModuleList([EncoderBlock(config)]*config.n_layer)
        self.decoder_p = nn.ModuleList([DecoderBlock(config)]*config.n_layer)

        # unfolding blocks
        self.unfold = nn.Unfold(kernel_size=(config.kernel_size,config.n_embd))
        # self.unfold_mask = nn.Unfold(kernel_size=(config.kernel_size,1))
        self.n_kmer = config.pep_block_size - config.kernel_size + 1 

        self.biD = 1  # 2 if bidirectional

        # linear mapping
        if config.select_kmer =='max':
            last_h = 1
        elif config.select_kmer == 'linear':
            last_h = 1
            self.score_head = nn.Linear(5, last_h)
        else:
            last_h = config.n_embd
            self.sel_attn = nn.MultiheadAttention(last_h, config.n_heads, config.attn_pdrop, batch_first=True)
            self.sel_head = nn.Linear(last_h, 1)
        
        self.head = nn.Sequential(nn.Linear(config.kernel_size*config.n_embd, 4*config.n_embd),
                                    nn.ReLU(),
                                    nn.Dropout(config.output_pdrop),
                                    nn.Linear(4*config.n_embd, config.n_embd),
                                    nn.ReLU(),
                                    nn.Linear(config.n_embd, last_h)
                                    )

        # loss
        # self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, p, p_mask, a, a_mask, k_mask, y=None, w=None, compile_att=False):
        """
        feed inputs to the model.
        Both input and output are provided as (batch, l_seq, feature).

        Parameters
        ----------
        p: torch.tensor, shape: b,lp
        p_mask: torch.tensor, shape: b,lp
        a: torch.tensor, shape: b,na,la
        a_mask: torch.tensor, shape: b,na
        k_mask: torch.tensor, shape: b,nk=lx-9+1=22
        y: torch.tensor, shape: b,ly

        """
        # peptide
        b, lp = p.size() 
        p = self.emb(p)     # b,lp,c
        p_pos_embeddings = self.pos_emb_p[:, :lp, :] # each position maps to a (learnable) vector
        p = self.drop_emb_p(p+p_pos_embeddings)  # b,lp,c
        
        # allele
        b, na, la = a.size()
        a = self.emb(a)     # b,na,la,c
        a_pos_embeddings = self.pos_emb_a[:, :, :la, :] # each position maps to a (learnable) vector
        a = self.drop_emb_a(a+a_pos_embeddings)  # b,na,la,c

        # encoder
        # for encoder_ in self.encoder_p:
        #     p, _ = encoder_(p, p_mask) # p: (b, lp, c)
        a = torch.flatten(a, start_dim=0, end_dim=1)  # b*na,la,c
        for encoder_ in self.encoder_a:
            a, _ = encoder_(a, None)  # a: (b*na, la, c)

        # decoder
        p = torch.tile(p.unsqueeze(dim=1), (1,na,1,1))  # b,na,lp,c
        p = torch.flatten(p, start_dim=0, end_dim=1)  # b*na,lp,c
        p_mask = torch.tile(p_mask.unsqueeze(dim=1), (1,na,1))  # b,na,lp
        p_mask = torch.flatten(p_mask, start_dim=0, end_dim=1)  # b*na,lp
        aws = []
        for decoder_ in self.decoder_p:
            p, aw = decoder_(p, p_mask, a, None)  # p: (b*na, lp, c)
            if compile_att:
                aw = aw.unsqueeze(dim=1)  # b*na, 1, lp, la
                aw = aw.view(b, na, 1, lp, la)
                aws.append(aw.cpu().detach().numpy())

        # unfold
        p = p.unsqueeze(dim=1) #b*na,1,lp,c
        p = self.unfold(p)   # b*na,k*c, nk
        p = p.transpose(-2,-1)  # b*na,nk, k*c
        assert p.shape[1] == self.n_kmer

        # mlp pred
        y_att = self.head(p)  # b*na,nk,1/h   (1 if max , h if attention selection)
        y_att = y_att.view(b,na,self.n_kmer)  # b, na, nk  

        # mask
        y_att = y_att.masked_fill(k_mask.unsqueeze(1), float('-inf'))  # b,na,nk
        y_att = y_att.masked_fill(a_mask.unsqueeze(2), float('-inf'))  # b,na,nk

        # select
        y_, _ = y_att.max(dim=2)  # b,na
        y_, _ = y_.max(dim=1)     # b,
        y_ = y_.unsqueeze(dim=1)    #b,1

        # print('\n', torch.sigmoid(y_[0]), '\n', torch.sigmoid(y_att[0]) )

        if y is not None:
            loss = self.bce_loss(y_, y)  # reduction='none' --> b,1
            loss = loss * w
            loss = torch.mean(loss)
            return y_, loss, y_att

        if compile_att:
            aws = np.concatenate(aws, axis=2)  # b, na, T=8, lp, la
            return y_, None, y_att, aws
                
        return y_, None, y_att   # b,1   # b, na, nk  
