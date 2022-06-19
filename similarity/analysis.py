import matplotlib.pyplot as plt
import torch 
import pandas as pd
import json
import ipdb
with open('/home/skku/vscode/ISE2/data/coco/annotations/captions_val2014.json') as f:
    json_data = json.load(f)

def cap_list(img_id):
    cap_list = []
    for ann in json_data['annotations']:
        if ann['image_id'] == img_id:
            cap_list.append(ann['caption']) 
    
    return cap_list


df = pd.read_csv('coco_val_sim.csv')

df['cap_similarity'].plot(kind='hist',bins = 100, figsize=(10,6), title='Distribution of similarity')
plt.show()

# new_dataframe.to_csv('coco_val_sim.csv')



