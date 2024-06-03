import torch
from PIL import Image
import numpy as np
import argparse
import random
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import csv

def get_score(stats_path):
    ops_positive = {
        'blurry_score' : 0.5,
        'dark_score' : 0.85,
        'grayscale_score' : 1,
        'image_text_similarity' : 0.25,
        'lang_score' : 0.5,
        'light_score' : 0.9,
        'low_information_score' : 0.9,
        'odd_aspect_ratio_score' : 0.8,
        # 'odd_size_score' : 0.9
    }
    ops_negative ={
        'char_rep_ratio' : 0.5,
        'word_rep_ratio' : 0.5
    }
    score_dict = {i : [] for i in {**ops_positive, **ops_negative}}
    all_score_vec = []

    with open(stats_path, 'r', encoding='utf-8') as jfile,\
            tqdm(total=sum(1 for _ in jfile)) as pbar:
        jfile.seek(0)

        for data in jfile:

            score_vec = []
            score_info = json.loads(data.strip())['__dj__stats__']

            for ops_name, score in score_info.items():
                if isinstance(score, list):
                    score=score[0]
                if ops_name in score_dict:
                    score_vec.append(score)
            all_score_vec.append(score_vec)
            pbar.update(1)
       
    return all_score_vec


def make_dataset(raw_jsonl,eval_jsonl,writein,cnt):

    print('making data')
    all_raw_score = get_score(raw_jsonl)
    print('finish loading raw')
    all_eval_score = get_score(eval_jsonl)
    print('finish loading eval')

    with open(writein, mode='w', newline='') as csv_file:  
        writer = csv.writer(csv_file)  
        selected_raw_score = random.choices(all_raw_score,k=cnt)
        selected_eval_score = random.choices(all_eval_score,k=cnt)     

        score = selected_raw_score+selected_eval_score
        label = [0]*cnt+[1]*cnt

        for i,j in zip(score,label):
            writer.writerow(i+[j])  


class StackingDataset(Dataset):

    def __init__(self, csv_file, train=True, split_ratio=0.8, transform=None):
        # transform if need
        self.transform = transform
        # load all data points
        self.data = []
        with open(csv_file, mode='r') as file:
            total_lines = sum(1 for _ in file)
            file.seek(0) 
            for row in tqdm(csv.reader(file), total=total_lines, desc="Loading data"):
                self.data.append([float(i) for i in row])
        random.shuffle(self.data)

    
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        
        vec = self.data[idx][0:10]
        label = self.data[idx][10]

        vec = torch.tensor(vec, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return vec,label


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    # data_dir
    parser.add_argument(
        "--raw_jsonl", type=str, 
        default='/mnt/share_disk/LIV/datacomp/processed_data/880w_1088w_dedup_processed/v0.1/combined_stats.jsonl', 
        help="jsonl file path of filtered results"
        )
    parser.add_argument(
        "--eval_jsonl", type=str, 
        default='/mnt/share_disk/LIV/datacomp/processed_data/evalset_processed/evalset_processed_stats.jsonl', 
        help="jsonl file path of filtered stats"
        )
    parser.add_argument(
        "--csv_path", type=str, 
        default='/root/GD-Based-Ensemble/data/stacking_data_60w.csv', 
        help="jsonl file path of filtered stats"
        )
    
    args = parser.parse_args()
    make_dataset(args.raw_jsonl,args.eval_jsonl,args.csv_path,600000)
