### prediction give by functional measures

import os, sys
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())

from utils.myutil import *
from unitmatch import read_MatchTable_from_csv

def filter_MatchTable_by_refpop(MatchTable):
    MatchTable = MatchTable[MatchTable['refPopRank'] == 1]
    MatchTable = MatchTable[MatchTable['RecSes1'] != MatchTable['RecSes2']]
    columns_to_keep = ['ID1', 'ID2', 'RecSes1', 'RecSes2', 'MatchProb', 'refPopCorr', 'refPopRank', 'refPopSig']
    MatchTable = MatchTable[columns_to_keep]
    return MatchTable

def filter_MatchTable_by_refpop2(MatchTable):
    # remain refPopRank ==1 and refPopRank == 2
    MatchTable = MatchTable[(MatchTable['refPopRank'] == 1) | (MatchTable['refPopRank'] == 2)]
    MatchTable = MatchTable[MatchTable['RecSes1'] != MatchTable['RecSes2']]
    columns_to_keep = ['ID1', 'ID2', 'RecSes1', 'RecSes2', 'MatchProb', 'refPopCorr', 'refPopRank', 'refPopSig']
    MatchTable = MatchTable[columns_to_keep]
    return MatchTable

def find_match_pairs_func(MatchTable,session_pair):
    pairs = []
    for index, row in MatchTable.iterrows():
        id1 = int(row['ID1'])
        id2 = int(row['ID2'])
        recses1 = row['RecSes1']
        recses2 = row['RecSes2']
        if recses1 == 1 and recses2 == 2:
            # Check if the reverse pair exists (cross validation)
            if MatchTable[(MatchTable['ID1'] == id2) & (MatchTable['ID2'] == id1) & (MatchTable['RecSes1'] == 2) & (MatchTable['RecSes2'] == 1)].any().any():
                pairs.append([id1, id2])
    pairs = np.array(pairs)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    np.save(os.path.join(results_save_folder,filename),pairs)
    return pairs

def find_match_pairs_func_Sig(MatchTable,session_pair):
    pairs = []
    for index, row in MatchTable.iterrows():
        id1 = int(row['ID1'])
        id2 = int(row['ID2'])
        recses1 = row['RecSes1']
        recses2 = row['RecSes2']
        refPopSig = row['refPopSig']
        if recses1 == 1 and recses2 == 2 and refPopSig == 1:
            # Check if the reverse pair exists (cross validation) and also has refPopSig == 1
            reverse_pair = MatchTable[(MatchTable['ID1'] == id2) & (MatchTable['ID2'] == id1) & (MatchTable['RecSes1'] == 2) & (MatchTable['RecSes2'] == 1) & (MatchTable['refPopSig'] == 1)]
            if reverse_pair.any().any():
                pairs.append([id1, id2])

    pairs = np.array(pairs)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func_Sig',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    np.save(os.path.join(results_save_folder,filename),pairs)
    return pairs

def find_match_pairs_func_Diff(MatchTable,session_pair):
    pairs = []
    for index, row in MatchTable.iterrows():
        id1 = int(row['ID1'])
        id2 = int(row['ID2'])
        recses1 = row['RecSes1']
        recses2 = row['RecSes2']
        refPopRank = row['refPopRank']
        if recses1 == 1 and recses2 == 2 and refPopRank == 1:
            # Find the reverse pair
            reverse_pair = MatchTable[(MatchTable['ID1'] == id2) & (MatchTable['ID2'] == id1) & 
                                      (MatchTable['RecSes1'] == 2) & (MatchTable['RecSes2'] == 1)]
            if reverse_pair.any().any():
                # Get refPopCorr for both ranks (1 and 2) for direct and reverse pairs
                corr_rank1_direct = row['refPopCorr']
                corr_rank1_reverse = reverse_pair['refPopCorr'].values[0]  # Assuming reverse_pair exists
                corr_rank2_direct = MatchTable[(MatchTable['ID1'] == id1) & (MatchTable['RecSes1'] == 1) & 
                                               (MatchTable['refPopRank'] == 2)]['refPopCorr'].mean()
                corr_rank2_reverse = MatchTable[(MatchTable['ID1'] == id2) & (MatchTable['RecSes1'] == 2) & 
                                                (MatchTable['refPopRank'] == 2)]['refPopCorr'].mean()
                mean_corr_rank1 = (corr_rank1_direct + corr_rank1_reverse) / 2
                mean_corr_rank2 = (corr_rank2_direct + corr_rank2_reverse) / 2
                # Check the difference in means
                if (mean_corr_rank1 - mean_corr_rank2) > 0.05:
                    pairs.append([id1, id2])
    pairs = np.array(pairs)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func_Diff',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    np.save(os.path.join(results_save_folder,filename),pairs)
    return pairs

if __name__ == '__main__':
    # train
    mode = 'test' # 'train' or 'test'
    # mouse = 'AV013'
    # probe = '19011119461'
    # location = '8'
    # dates = ['2022-06-09', '2022-06-10']
    # exps = ['AV013_2022-06-09_ActivePassive_g0_t0-imec0-ap', 
    #         'AV013_2022-06-10_ActivePassive_g0_t0-imec0-ap']
    # session_pair = '2'
    mouse = 'AL036'
    probe = '19011116882'
    location = '3'
    dates = ['2020-02-24', '2020-02-25']
    exps = ['AL036_2020-02-24_stripe240_NatIm_g0_t0-imec0-ap', 
            'AL036_2020-02-25_stripe240_NatIm_g0_t0-imec0-ap']
    session_pair = '1'
    print('mouse', mouse, 'session_pair', session_pair)
    MatchTable = read_MatchTable_from_csv(mouse, session_pair)
    # # method 1 and 2, which don't use rank 2 information
    # MatchTable = filter_MatchTable_by_refpop(MatchTable)
    # match_pair = find_match_pairs_func(MatchTable,session_pair)
    # match_pair = find_match_pairs_func_Sig(MatchTable,session_pair)

    # method 3, which use rank 2 information
    MatchTable = filter_MatchTable_by_refpop2(MatchTable)
    match_pair = find_match_pairs_func_Diff(MatchTable,session_pair)

    # # save the filtered MatchTable to csv
    # results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func',mouse)
    # if not os.path.exists(results_save_folder):
    #     os.makedirs(results_save_folder)
    # filename = 'MatchTable_'+ 'session_pair' + str(session_pair) + '.csv'
    # MatchTable.to_csv(os.path.join(results_save_folder,filename),index=False)

    has_conflict_match(match_pair[:,0])
    has_conflict_match(match_pair[:,1])    

    match_pair = match_pair.tolist()
    print('match_pair', match_pair)
    print('len', len(match_pair))
     