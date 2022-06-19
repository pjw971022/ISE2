from __future__ import annotations
from audioop import avg
import fiftyone as fo
import fiftyone.zoo as foz
import json
import ipdb
from regex import F
from wandb import set_trace
import pandas as pd
import os
#caption visualize 
with open('/home/skku/vscode/ISE2/data/coco/annotations/captions_val2014.json') as f:
    json_data = json.load(f)

def cap_list(img_id):
    cap_list = []
    for ann in json_data['annotations']:
        if ann['image_id'] == img_id:
            cap_list.append(ann['caption']) 
    
    return cap_list

from transformers import AlbertTokenizer, AlbertModel
import torch
import random
from tqdm import tqdm
from glob import glob
model = None
tokenizer = None
from sentence_transformers import SentenceTransformer, util

# model_origin = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def cosim_sentence(st1, st2):
 
    #Compute embedding for both lists
    embedding_1 = model.encode(st1, convert_to_tensor=True,show_progress_bar =False)
    embedding_2 = model.encode(st2, convert_to_tensor=True,show_progress_bar =False)
    
    cosim = util.pytorch_cos_sim(embedding_1, embedding_2).squeeze(0).squeeze(0)
    

    return cosim.item()

if __name__ == '__main__':
    images = [f for f in glob(f'/home/skku/vscode/ISE2/data/coco/images/val2014/*')]
    random.shuffle(images)
    avg_sim = 0.0
    max_sim = 0.0
    min_sim = 1.0
    sim_list = []
    for idx, file in enumerate(tqdm(images,desc="caption_sim",mininterval=0.1)):    
        img_id = int(file.split('/')[-1].split('_')[-1].replace('.jpg','').lstrip('0'))
        clist = cap_list(img_id)
        total_cap_sim = 0.0
        cnt = 0
        for cap1 in clist:
            for cap2 in clist:
                if cap1 != cap2:
                    total_cap_sim += cosim_sentence(cap1, cap2)
                    cnt += 1
        if cnt==0:
            print(cnt, img_id, clist)
            continue
        total_cap_sim /= cnt
        avg_sim += total_cap_sim
          
        max_sim = max(max_sim,total_cap_sim)
        min_sim = min(min_sim,total_cap_sim)
        # sim_list.append(total_cap_sim)      
        sim_list.append({'img_id':img_id, 'similarity':total_cap_sim }) 
        if total_cap_sim >1.0:
            print(f"image_id:{img_id} caption_similarity:{total_cap_sim} cnt: {cnt}")
        
        if idx%10000 == 0:        
            print(f"max_similarity: {max_sim}   min_similarity: {min_sim}")        
            print(f"avg similarity:{avg_sim/(idx+1)}")

            
    CUR_DIR = os.getcwd()
    df = pd.DataFrame(sim_list)
    df.to_csv(CUR_DIR + '/similarity/coco_val_sim.csv',header=False,index=False)