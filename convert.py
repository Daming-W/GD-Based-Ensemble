import os 
import json
import jsonlines
from tqdm import tqdm
import numpy as np
import concurrent
from concurrent.futures import ThreadPoolExecutor

# convert processed jsonl to datacomp npy for training/evaluating
def dj2dc_writer(jsonl_path, npy_path):

    # Preload all lines to avoid issues related to concurrent file reading 
    with jsonlines.open(jsonl_path, 'r') as jsonlfile:
        lines = list(jsonlfile)
    # print(lines)
    total_jobs = len(lines)
    uid_list = []

    def process_line(line):
        # get corresponding json file
        json_path = os.path.join(line['images'][0].replace('.jpg', '.json'))
        
        if os.path.exists(json_path):
            # get uid from json
            with open(json_path, 'r') as j:
                data = json.load(j)
                # print(data['uid'])
                return data['uid']

    with ThreadPoolExecutor() as executor:
        # Using tqdm to show progress bar in conjunction with futures
        futures = [executor.submit(process_line, line) for line in lines]
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_jobs):
            uid_list.append(future.result())

    # convert to u8 and save npy
    processed_uids = np.array(
                            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uid_list], 
                            np.dtype("u8,u8")
                            )
    processed_uids.sort()
    
    if npy_path:
        np.save(npy_path, processed_uids)
        print(f'saving npy file >> {npy_path}')
    else:
        print('skip saving npy file') 
    return uid_list

dj2dc_writer(
    '/mnt/share_disk/LIV/datacomp/ensemble/880w_importance_ensemble.jsonl',
    '/mnt/share_disk/LIV/datacomp/ensemble/880w_importance_ensemble.npy'
)