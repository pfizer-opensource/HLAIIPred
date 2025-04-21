import os
import numpy as np
import pandas as pd
import torch
from hlapred.predict import HLAIIPredict 

def load_dummy_data(alleles='9common'):
    peptides = ['IKKWEKQVSQKKKQKN', 'KANVKIFKSQGAA']

    # alleles
    if alleles == '9common':
        alleles = ['DRB1*01:01', 'DRB1*03:01', 'DRB1*04:01', 'DRB1*07:01', 'DRB1*08:01', 'DRB1*09:01', 'DRB1*11:01', 'DRB1*13:01', 'DRB1*15:01',0,0,0]
        alleles = [alleles]*len(peptides)
    else:
        alleles += [0]*(14-len(alleles))
        alleles = [alleles] * len(peptides)

    return peptides, alleles


############################################

model_root = 'models'
device = torch.device('cpu')  # cpu or cuda:0
out_root = 'predictions/'
alleles = '9common'
# alleles = ['DRB1*14:54','DPA1*01:03-DPB1*02:01','DQA1*01:02-DQB1*06:02']

peptides, alleles = load_dummy_data(alleles) 

if not os.path.exists(out_root):
    os.makedirs(out_root)

df = pd.DataFrame()
df['peptide'] = peptides

for model_idx in range(2):
    predictor = HLAIIPredict(model_root, model_idx, device)
    inputs = predictor.prepare_input(peptides, alleles)
    y_pred, scores = predictor.predict(inputs, batch_size=300, sigmoid=True)
    # np.save(os.path.join(out_root, 'model_%i_pred.npy'%(model_idx)), y_pred)   # y_pred: the best score among all 9mers and alleles
    np.save(os.path.join(out_root, 'model%i_matrix.npy'%(model_idx)), scores)   # scores: all 9mer and allele scores  # shape: n_peptides, n_alleles, n_9mers
    df['model_%i'%model_idx] = y_pred
    
df['prediction'] = df[['model_%i'%model_idx for model_idx in range(2)]].mean(axis=1)
df.to_csv(os.path.join(out_root, 'predicted_scores.csv'), index=False)



