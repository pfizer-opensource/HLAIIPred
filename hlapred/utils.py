import sys
import numpy as np
import torch
import random
import pandas as pd


SEQUENCE_KEY = 'peptide'
LABEL_KEY = 'presenter'
WEIGHT_KEY = 'weight'
MAX_N_ALLELES = 14
L_ALPHA_CHAIN = 15
L_BETA_CHAIN = 19
BLOCK_SIZE_PEP = 30
KMER = 9

# AA_vocab and pos_vocab were used to train activity oracle
AA_map = {'<PAD>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11,'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 0, 'U': 0, 
            '.':21, '>':22, '<start>': 23, '<stop>':24}

AA_map_reversed = {0: '<PAD>', 1: 'A', 2: 'C', 3: 'D',
                     4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                     9: 'K', 10: 'L', 11: 'M', 12: 'N',
                     13: 'P', 14: 'Q', 15: 'R', 16: 'S',
                     17: 'T', 18: 'V', 19: 'W', 20: 'Y',
                     21: '.', 22:'>', 23:'<start>', 24:'<stop>'}

VOCAB_SIZE = len(AA_map_reversed)

def consecutive_numbers(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    l = list(zip(edges, edges))
    return [range(i[0],i[1]+1) for i in l]

def get_encoding(df):
    df.index = range(df.shape[0])
    input_encoding = np.zeros([len(df), BLOCK_SIZE_PEP])
    input_mask = np.zeros_like(input_encoding)
    kmer_mask = np.zeros([len(df), BLOCK_SIZE_PEP-KMER+1])
    allele_encoding = np.zeros([len(df), MAX_N_ALLELES, L_ALPHA_CHAIN+1+L_BETA_CHAIN])
    allele_mask = np.zeros([len(df), MAX_N_ALLELES])

    for idx, row in df.iterrows():
        # seq
        seq = row[SEQUENCE_KEY]
        seq_enc = [AA_map[aa] for aa in seq]
        input_encoding[idx, :len(seq)] = seq_enc
        input_mask[idx, :len(seq)] = [1] * len(seq)
        if len(seq) == 8:
            m = 1
        else:
            m = len(seq) - KMER + 1
        kmer_mask[idx, :m] = [1] * m

        # presenter & weight
        presenter = df[LABEL_KEY].values.reshape(-1,1)
        weight = df[WEIGHT_KEY].values.reshape(-1,1)

        # allele
        for ia, al in enumerate(['allele%i'%i for i in range(1,MAX_N_ALLELES+1)]):
            al_enc = [AA_map[aa] for aa in row[al]]
            allele_encoding[idx,ia,:] = al_enc
            if sum(al_enc[:5]) + sum(al_enc[-5:]) == 0:
                allele_mask[idx,ia] = 1


    input_mask = 1 - input_mask         # padded elements are masked with 1
    kmer_mask = 1 - kmer_mask         # padded elements are masked with 1


    return input_encoding, input_mask, kmer_mask, allele_encoding, allele_mask, presenter, weight


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
