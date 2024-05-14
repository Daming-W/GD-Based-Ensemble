import numpy as np  
from sklearn.manifold import TSNE  
import umap
import matplotlib.pyplot as plt  
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from umap import UMAP


raw_emb_path = '/mnt/share_disk/LIV/datacomp/processed_data/1088w_emb/1088w_emb_stats.jsonl'
eval_emb_path = '/mnt/share_disk/LIV/datacomp/processed_data/evalset_emb/evalset_emb_stats.jsonl'
raw_emb = eval_emb = []

def process_line(line):
    info = json.loads(line)
    return info['__dj__stats__']['image_embedding'][0]

lines=[]
with open(raw_emb_path, 'r') as r:
    for i in range(30000):
        line = r.readline()
        if not line:
            break
        lines.append(line)

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_line, line) for line in tqdm(lines, desc="Processing lines")]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Completing tasks"):
        raw_emb.append(future.result())

lines=[]
with open(eval_emb_path, 'r') as r:
    for i in range(30000):
        line = r.readline()
        if not line:
            break
        lines.append(line)

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_line, line) for line in tqdm(lines, desc="Processing lines")]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Completing tasks"):
        eval_emb.append(future.result())

umap_model = UMAP(n_components=2) 
 
two_d_embeddings1 = umap_model.fit_transform(eval_emb) 
two_d_embeddings2 = umap_model.transform(raw_emb)  

plt.figure(figsize=(12, 10))  
scatter1 = plt.scatter(two_d_embeddings1[:, 0], two_d_embeddings1[:, 1], label='raw', c='blue',alpha=0.5)  
scatter2 = plt.scatter(two_d_embeddings2[:, 0], two_d_embeddings2[:, 1], label='eval', c='red',alpha=0.5) 
plt.title('umap visualization and comparison of two datasets')  
plt.xlabel('umap feature 1')  
plt.ylabel('umap feature 2')  
plt.legend()
plt.savefig('test.png')