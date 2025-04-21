import argparse
import glob
import os
import yaml
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from Bio import SeqIO

from hlapred.model_modules import TransformerConfig, DeConvolutionalAttention
from hlapred.dataset import AntibodyDataset
from hlapred.utils import SEQUENCE_KEY, LABEL_KEY, BLOCK_SIZE_PEP, MAX_N_ALLELES, L_ALPHA_CHAIN, L_BETA_CHAIN, WEIGHT_KEY

class HLAIIPredict(object):
    """
    This class gets a protein sequence and using a sliding window returns a heatmap and corresponding
    evaluation metrics.
    """
    def __init__(self, model_root, fold_idx, device, mhc2_root=None):
        
        self.device = device
        settings_path = os.path.join(model_root, 'config.yaml')
        settings = yaml.safe_load(open(settings_path, "r"))
        
        # Presentation model
        mconf = TransformerConfig(**settings['model'])
        model = DeConvolutionalAttention(mconf)
        state_dict = torch.load(os.path.join(model_root, 'epT_%i.pt'%fold_idx), map_location=device)
        if 'MODEL_STATE' in state_dict.keys():
            state_dict = state_dict['MODEL_STATE']
        model.load_state_dict(state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        model.to(device)

        self.model = model
        self.model.eval()

        # allele meta
        if mhc2_root is None:
            mhc2_root = "."
        self.alpha_dict = OrderedDict([(f.id, str(f.seq)) for f in list(SeqIO.parse(os.path.join(mhc2_root, "DRA_DPA_DQA_pseudosequence.fasta"), 'fasta'))])
        self.beta_dict = OrderedDict([(f.id, str(f.seq)) for f in list(SeqIO.parse(os.path.join(mhc2_root, "DRB_DPB_DQB_pseudosequence.fasta"), 'fasta'))])


    def prepare_input(self, peptides, alleles):
        """
        peptides: list 
            contains peptide strings
        alleles: list 
            contains one allele set for each peptide peptide. Each allele set contains <=14 allele names like: DRA*01:01-DRB7*01:01
        """
        assert len(peptides) == len(alleles), 'The number pf peptides and alleles must be equal.'

        df = pd.DataFrame()
        assert np.array([len(p)<=BLOCK_SIZE_PEP for p in peptides]).all(), 'The length of peptides must not exceed %i.'%BLOCK_SIZE_PEP
        df[SEQUENCE_KEY] = peptides
        df[LABEL_KEY] = [0]*df.shape[0]
        df[WEIGHT_KEY] = [1]*df.shape[0]

        # allele

        # check number of alleles and pad with 0 to the max number
        assert np.array([len(a)<=MAX_N_ALLELES for a in alleles]).all(), 'Number of alleles per peptide must not exceed %i.'%MAX_N_ALLELES
        alleles = np.array([list(a)+[0]*int(MAX_N_ALLELES-len(a)) for a in alleles])

        def allelize(x):
            if x in [0,'0']:
                return 'X'*L_ALPHA_CHAIN + '.' + 'X'*L_BETA_CHAIN
            elif '-' in x:
                a,b = x.split('-')
                if a == 'DRA*01:01':
                    a = 'DRA*01:01:01:01'
            elif 'DRB' in x:
                a = 'DRA*01:01:01:01'
                b = x
            else:
                raise IOError('allele is unknown: %s'%x)
            
            alpha_ps = self.alpha_dict[a]
            beta_ps = self.beta_dict[b]

            return alpha_ps+'.'+beta_ps
        
        al_df = pd.DataFrame(alleles, columns=['allele%i'%(i+1) for i in range(MAX_N_ALLELES)])
        assert al_df.shape[1] == MAX_N_ALLELES

        for al_col in al_df.columns:
            df[al_col] = al_df[al_col].apply(lambda x: allelize(x)) 

        return df

    def predict(self, df, batch_size, sigmoid=True, compile_att=False):
        dataset = AntibodyDataset(df)
        loader = DataLoader(dataset, shuffle=False, pin_memory=True,batch_size=batch_size,num_workers=1)
        pbar = tqdm(enumerate(loader), total=len(loader))
        y_pred = []
        att_matrics = []
        cross_att_weights = []
        for it, d_it in pbar:
            p, p_mask, k_mask, a, a_mask, y, w = d_it

            # place data on the correct device
            p = p.to(self.device)
            p_mask = p_mask.to(self.device)
            k_mask = k_mask.to(self.device)
            a = a.to(self.device)
            a_mask = a_mask.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(False):
                if compile_att:
                    y_, loss, att, aws = self.model(p, p_mask, a, a_mask, k_mask, y=None, compile_att=compile_att)
                    cross_att_weights.append(aws)
                else:
                    y_, loss, att = self.model(p, p_mask, a, a_mask, k_mask, y=None)  # b,1  # b,na,nk
            
            # sigmoid
            if sigmoid:
                y_ = torch.sigmoid(y_)
                att = torch.sigmoid(att)

            y_pred.append(y_.detach().cpu().numpy())
            att_matrics.append(att.detach().cpu().numpy())

        y_pred = np.concatenate(y_pred)
        att_matrics = np.concatenate(att_matrics)
        
        if compile_att:
            cross_att_weights = np.concatenate(cross_att_weights)   # b,na,T=8,lp,la
            return y_pred, att_matrics, cross_att_weights  # b,1   # b,na,nk   # b,na,T,lp,la
                
        return y_pred, att_matrics  # b,1   # b,na,nk


    


