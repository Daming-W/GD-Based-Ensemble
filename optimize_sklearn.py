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
    
    df = init_df(args.filter_res_jsonl, args.filter_stats_jsonl)
