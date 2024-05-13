import os
import json
import jsonlines
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_reader import init_df,filter_by_bool,weighted_vote_ensemble,filter_by_threshold

# loss
class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()
    
    def forward(self, output1, output2):
        return F.pairwise_distance(output1, output2).pow(2).mean()

# compute filtered image emb distribution for loss cal
def compute_emb_distribution(emb):

    emb_mean = np.mean(emb, axis=0)
    emb_var = np.var(emb, axis=0)

    return [emb_mean,emb_var]

# get emb from filtered index
def load_emb(df, index):

    return df[df['image_embedding'].isin(index)]  

# optimize weights per iteration
def optimize_iter(df, emb_dis_eval, weights, criterion, optimizer):
    
    weights = F.softmax(weights, dim=0)  
    
    optimizer.zero_grad()

    # filtered_index = weighted_vote_ensemble(score_df, weights, jsonl_path, threshold=0.5)
    index = weighted_vote_ensemble(df, weights.tolist(), None, 0.5)
    
    # get filtered image emb 
    emb = load_emb(df,index)

    # get emb dis for filtered data and eval data
    emb_dis = compute_emb_distribution(emb)
    
    # cal loss and backward
    loss = criterion(emb_dis, emb_dis_eval)

    loss.backward()
    print(weights.grad)
    optimizer.step()

    return weights, loss.item()


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # data_dir
    parser.add_argument("--filter_res_jsonl", type=str, 
                        default='/mnt/share_disk/LIV/datacomp/processed_data/1088w_emb.jsonl', 
                        help="jsonl file path of filtered results")
    parser.add_argument("--filter_stats_jsonl", type=str, 
                        default='/mnt/share_disk/LIV/datacomp/processed_data/1088w_emb_stats.jsonl', 
                        help="jsonl file path of filtered stats")
    # param
    parser.add_argument("--num_ops", type=int, default=3, help="operator number")
    # optimize
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs.")

    args = parser.parse_args()

    torch.manual_seed(1)

    df = init_df(args.filter_res_jsonl, args.filter_stats_jsonl)
    df = filter_by_threshold(df)
    print(df)
    
    # weights init

    # weights = np.array([1/args.num_ops] *  args.num_ops)
    # weights = torch.nn.Parameter(torch.ones(args.num_ops) / args.num_ops, requires_grad=True)

    weights = torch.nn.Parameter(torch.randn(args.num_ops), requires_grad=True)
    weights = F.softmax(weights, dim=0)  
    print(f'init weights -> {weights}')

    # load eval_emb
    eval_dataset_size = 700000
    eval_emb = load_emb(df,list(range(eval_dataset_size + 1)))
    
    # compute eval embedding distribution as gt
    emb_dis_eval = compute_emb_distribution(eval_emb)
    emb_dis_eval = [1,1]
    # get criterion
    criterion = EuclideanDistanceLoss()

    # get optimizer
    optimizer = torch.optim.SGD([weights], lr=0.01)

    # TRAIN
    with tqdm(total=len(args.num_epochs)) as pbar:
        for _ in range(args.num_epochs):
            weights, loss = optimize_iter(df, emb_dis_eval, weights, criterion, optimizer)
            pbar.set_description('optimizing')
            pbar.set_postfix({'loss':loss,'weights':weights})
            pbar.update(1) 

            