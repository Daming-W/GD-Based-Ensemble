import torch
import argparse
from tqdm import tqdm
import jsonlines
import numpy as np

from utils.dataset import get_score
from LR_fit import LogisticRegressionModel
from utils.engine import train_epoch,eval_epoch

def inference(model, data, output_file, start_index):

    model.eval()
    scores = []
    for i in data:
        i=torch.tensor(i)
        pred = model.predict(i)
        print(pred)
        scores.append(pred)

    with open(output_file, 'a') as file:
        for index, score in enumerate(scores, start=start_index):
            file.write(f"{score}\n")

if __name__ == "__main__":
    print('inf')
    parser = argparse.ArgumentParser()
    # path and dir
    parser.add_argument(
        "--model_save_path", type=str, 
        default='/root/GD-Based-Ensemble/checkpoints/0514_1232.pth'
        )
    # device
    parser.add_argument(
        "--gpu_id", type=str, 
        default='cuda',
        help="GPU id to work on, \'cpu\'."
        )
    # data
    parser.add_argument(
        "--raw_jsonl", type=str, 
        default='/mnt/share_disk/LIV/datacomp/processed_data/880w_1088w_dedup_processed/v0.1/combined_stats.jsonl', 
        help="jsonl file path of filtered results"
    )
    parser.add_argument(
        "--res_path", type=str,
        default='/root/GD-Based-Ensemble/results/lr1.txt', 
        help="output"
        )   
    args = parser.parse_args()

    # set model
    model = LogisticRegressionModel(10)
    model.load_state_dict(torch.load(args.model_save_path)['model_state_dict'])

    print('model ready to go')

    all_raw_score = get_score(args.raw_jsonl)

    inference(model,all_raw_score,args.res_path,0)