import os 
import json
import jsonlines
import tarfile
import argparse
import glob
from tqdm import tqdm
import numpy as np
import concurrent
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,as_completed
from convert import *

def process_line(line, is_stats=False):
    infos = json.loads(line)
    if is_stats:
        infos = infos['__dj__stats__']
        infos = {key: value[0] if isinstance(value, list) else value for key, value in infos.items()}
    else:
        infos = infos['images'][0]
    return infos

def init_df(raw_jsonl_path, stats_jsonl_path):

    with open(raw_jsonl_path, 'r') as reader:
        lines = reader.readlines()
    
    print('start loading img path')
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_line, line) for line in lines]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            results.append(future.result())
    df = pd.DataFrame(results, columns=['images'])

    if stats_jsonl_path:

        print('start loading stats')

        with open(stats_jsonl_path, 'r') as reader:
            lines = reader.readlines()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_line, line, is_stats=True) for line in lines]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Stats"):
                results.append(future.result())
        score_df = pd.DataFrame(results)

        merged_df = pd.concat([df, score_df], axis=1)
        print(f'(N-samples, N-ops) -> {merged_df.shape[0], merged_df.shape[1]-1} \n')
        print(merged_df)
        return merged_df
    
    else:
        print(f'Read raw image paths only with {len(results)} images processed')
        return df


# load ops score/results to df
# 用于使用筛选后的jsonl对全量进行过滤
def filter_by_bool(df, jsonl_dir):
    
    # read keep data sample index
    for jsonl_file in glob.glob(jsonl_dir+'/*.jsonl'):
        if 'stat' not in jsonl_file:
            keep_id = []
            with open(jsonl_file,'r') as reader:
                print(f'reading from {jsonl_file}\n')
                op_name = os.path.splitext(os.path.basename(jsonl_file))[0]
                for line in reader:
                    infos = json.loads(line)
                    keep_id.append(infos['images'][0])
            # make 1 for kept, make 0 for filtered
            keep_set = set(keep_id)
            flag = [1 if i in keep_set else 0 for i in df['images']]
            # append to df
            df[op_name]= flag
            print(f'{op_name} filter keeps {len(keep_id)} data samples\n')

    return df

# weighted vote method
def weighted_vote_ensemble(df, weights, jsonl_path, threshold=0.5):

    # if weights is None, then all ops worth equally
    if weights == None:
        weights = [1/(df.shape[1]-1)] * (df.shape[1]-1)

    # check shape matched and sum is 1
    assert len(weights) == df.shape[1]-1,'weights length should be equal to number of operators!!!'
    assert np.isclose(sum(weights), 1.0), 'weights sum should be 1.00'

    # do weighted sum and create a new col
    df['weighted_sum'] = df.apply(
        lambda row: sum(row[col] * weight for col, weight in zip(df.columns[1:], weights)), 
        axis=1)

    # filter by threshold
    filtered_index = df[df['weighted_sum'] > threshold]['images']
    print(f'output {len(filtered_index)} data samples')

    # write jsonl for conversion
    if jsonl_path:
        with open(jsonl_path,'w') as writer:
            for i in filtered_index:
                info = {'images':[i]}
                writer.write(json.dumps(info)+'\n')
        print(f'write jsonl >> {jsonl_path}')

    return filtered_index


# 
def filter_by_threshold(args, df):

    ops = {
        'blurry_score':0.5, 
        'char_rep_ratio':None, 
        'dark_score':0.5, 
        'grayscale_score':0.5,  
        'image_text_similarity':0.3, 
        'lang':'en', 
        'lang_score':0.2, 
        'light_score':None,
        'low_information_score':None, 
        'odd_aspect_ratio_score':None, 
        'odd_size_score':None, 
        'word_rep_ratio':None
        }
    
    used_ops = len({k: v for k, v in ops.items() if v is not None})
    
    ops_weights = {
        'blurry_score':1/used_ops, 
        'char_rep_ratio':1/used_ops, 
        'dark_score':1/used_ops, 
        'grayscale_score':1/used_ops,  
        'image_text_similarity':1/used_ops, 
        'lang':1/used_ops, 
        'lang_score':1/used_ops, 
        'light_score':1/used_ops,
        'low_information_score':1/used_ops, 
        'odd_aspect_ratio_score':1/used_ops, 
        'odd_size_score':1/used_ops, 
        'word_rep_ratio':1/used_ops
        }

    # init score col
    df['score'] = 0

    # loop filter by ops dict
    orig_shape = df.shape
    for op,th in ops.items():
        if th is not None:
            # filter by the score
            if type(th) is float:
                id = df[df[op] > th].index
                df.loc[id, 'score'] += 1*ops_weights[op]

            # filter by the language
            if type(th) is str:
                id = df[df[op]=='en'].index
                df.loc[id, 'score'] += 1*ops_weights[op]

            print(f'filtering by {op} with th={th}')

    print(orig_shape,' >> ',df.shape)   

    df = df[df['score']>0.5]

    return df

def write_jsonl(df, jsonl):
    with open(jsonl,'w') as writer:
        for i in df['images']:
            info = json.dumps({'images':[i]})+'\n'
            writer.write(info)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ops", type=int, default=10, help="number of all operators")
    args = parser.parse_args()

    raw_jsonl_path = '/mnt/share_disk/LIV/datacomp/processed_data/demo-vaquitai-10w/demo-processed.jsonl'
    stats_jsonl_path = '/mnt/share_disk/LIV/datacomp/processed_data/demo-vaquitai-10w/demo-processed_stats.jsonl'

    # raw_jsonl_path = '/mnt/share_disk/LIV/datacomp/processed_data/880w_1088w_dedup_processed/v0.1/demo-processed.jsonl'
    # stats_jsonl_path = '/mnt/share_disk/LIV/datacomp/processed_data/880w_1088w_dedup_processed/v0.1/combined_stats.jsonl'

    merged_df = init_df(raw_jsonl_path, stats_jsonl_path)

    df = filter_by_threshold(args, merged_df)

    write_jsonl(df, '/mnt/share_disk/LIV/datacomp/ensemble/880w_importance_ensemble.jsonl')

    dj2dc_writer(
        '/mnt/share_disk/LIV/datacomp/ensemble/880w_importance_ensemble.jsonl',
        '/mnt/share_disk/LIV/datacomp/ensemble/880w_importance_ensemble.npy'
    )