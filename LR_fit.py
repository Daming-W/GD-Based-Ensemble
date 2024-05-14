import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.dataset import get_score,StackingDataset
from utils.engine import train_epoch,eval_epoch
from utils.logger import Logger

def generate_default_logger_name():
    now = datetime.datetime.now()
    return now.strftime('%m%d_%H%M')

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
    "--csv_file", type=str, 
    default='/root/GD-Based-Ensemble/data/stacking_data_60w.csv', 
    help="data saved path"
    )
parser.add_argument(
    "--batch_size", type=int,
    default=128, 
    help="batch size of data"
    )      
parser.add_argument(
    "--num_workers", type=int,
    default=64, 
    help="number of workers"
    )   
parser.add_argument(
    "--model_save_path", type=str, 
    default='/root/GD-Based-Ensemble/checkpoints/'+generate_default_logger_name()+'.pth'
    )
parser.add_argument(
    "--logger_name", type=str, 
    default='/root/GD-Based-Ensemble/logger/log_'+generate_default_logger_name()
    )
parser.add_argument(
    "--epochs", type=int, 
    default=10, 
    help="total number of training epochs"
    )
args = parser.parse_args()

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
    

#load data
train_dataset = StackingDataset(csv_file=args.csv_file, train=True, split_ratio=0.8, transform=None)
val_dataset = StackingDataset(csv_file=args.csv_file, train=False, split_ratio=0.8, transform=None)

print(len(train_dataset))
print(len(val_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

model = LogisticRegressionModel(input_dim=10).to('cuda')

criterion = torch.nn.BCELoss().to('cuda')
optimizer = optim.SGD(model.parameters(), lr=0.01) 

# setup logger
logger = Logger(args.logger_name,False)
logger.append(args)
print('finish setting logger')

# train and eval
print('<<< start training >>>')
for epoch in range(args.epochs):
    print(f'<<< epoch {epoch+1} >>>')
    logger.append(f'epoch : {epoch+1}')
    train_epoch(args, train_dataloader, model, criterion, optimizer, logger)
    eval_epoch(args, val_dataloader, model, criterion, logger)

torch.save({
    'model_state_dict': model.state_dict()
}, args.model_save_path)