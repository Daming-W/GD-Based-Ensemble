import torch
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import torch.nn.functional as F


def train_epoch(args, dataloader, model, criterion, optimizer, logger):

    total_loss,total_acc = [],[]
    model.train()
    with tqdm(total=len(dataloader)) as pbar: 
        for i, (vec, label) in enumerate(dataloader):
            # load to device
            vec = vec.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # zero grad
            optimizer.zero_grad()
            # forward
            pred = model(vec)
            # compute loss
            loss = criterion(pred.squeeze(),label)
            # loss backward
            loss.backward()
            # optimizer and scheduler step
            optimizer.step()
            # sum losses in an epoch
            total_loss.append(loss.detach().cpu())
            # get acc

            # update pbar
            pbar.set_description('training')
            pbar.set_postfix({
                'loss(iter)': float(loss.detach().cpu()),
                'loss(mean)': np.mean(total_loss),
                # 'acc(iter)': float(acc.detach().cpu()),
                'acc(mean)': np.mean(total_acc)
            })
            pbar.update(1)

    epoch_loss = np.mean(total_loss)
    epoch_acc = np.mean(total_acc)

    if logger:
        logger.append(f'Train epoch loss: {epoch_loss}, Train epoch accuracy: {epoch_acc}')
    

def eval_epoch(args, dataloader, model, criterion, logger=None):
    total_loss = []
    total_acc = []  # List to store all accuracies for calculating mean accuracy
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for i, (vec, label) in enumerate(dataloader):
            # load to device
            vec = vec.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # forward
            pred = model(vec)
            # compute loss
            loss = criterion(pred.squeeze(),label)
            total_loss.append(loss.item())
            # Update progress bar
            pbar.set_description('evaluating')
            pbar.set_postfix({
                'loss(iter)': loss.item(),
                'loss(mean)': np.mean(total_loss),
                # 'acc(iter)': acc.item(),
                'acc(mean)': np.mean(total_acc)
            })
            pbar.update(1)

    epoch_loss = np.mean(total_loss)
    epoch_acc = np.mean(total_acc)
    # Log epoch level metrics
    if logger:
        logger.append(f'Eval epoch loss: {epoch_loss}, Eval epoch accuracy: {epoch_acc}')

