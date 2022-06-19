import fiftyone as fo
import json
from glob import glob
import ipdb
import random
from tqdm import tqdm
dataset_types = ['train','val'] #train/val/test
json_data1 = None
json_data2 = None

def cap_list(img_id):
    cap_list = []
    for ann in json_data1['annotations']:
        if ann['image_id'] == img_id:
            cap_list.append(ann['caption']) 
    
    return cap_list

def get_widthANDheight(img_id):
    width = None
    height = None
    for img_ann in json_data2['images']:
        if img_ann["id"]==img_id:
            width = img_ann["width"]
            height = img_ann["height"]
            break
    if width==None:
        ipdb.set_trace()
    return width, height


def bbox_list(img_id):
    """ 
        bounding_box (None): a list of relative bounding box coordinates in
            ``[0, 1]`` in the following format::

            [<top-left-x>, <top-left-y>, <width>, <height>]
    """
    bbox_list = []
    for ann in json_data2['annotations']:
        bbox = []

        if ann['image_id'] == img_id:
            width, height = get_widthANDheight(img_id)
            bbox.append(ann['bbox'][0]/width)
            bbox.append(ann['bbox'][1]/height)
            bbox.append(ann['bbox'][2]/width)
            bbox.append(ann['bbox'][3]/height)
            bbox_list.append(bbox)     

    return bbox_list




if __name__ == '__main__':
    # try:
    #     fo.load_dataset(f"{dataset_type}_dataset").delete()
    #     dataset1 = fo.Dataset(f"{dataset_type}_dataset")
    # except:
    #     dataset1 = fo.Dataset(f"{dataset_type}_dataset")

    try:
        fo.load_dataset(f"All_dataset").delete()
        dataset1 = fo.Dataset(f"All_dataset")
    except:
        dataset1 = fo.Dataset(f"All_dataset")

    for dataset_type in dataset_types:
        with open(f'/home/skku/vscode/ISE2/data/coco/annotations/captions_{dataset_type}2014.json') as f:
            json_data1 = json.load(f)

        with open(f'/home/skku/vscode/ISE2/data/coco/annotations/instances_{dataset_type}2014.json') as f:
            json_data2 = json.load(f)

        images = [f for f in glob(f'/home/skku/vscode/ISE2/data/coco/images/{dataset_type}2014/*')]
        random.shuffle(images)

        for idx, file in enumerate(tqdm(images,desc="sampling",mininterval=0.1)):    
            sample = fo.Sample(filepath=file, tags=[f'{dataset_type}'])
            img_id = int(file.split('/')[-1].split('_')[-1].replace('.jpg','').lstrip('0'))
            sample["captions"] = cap_list(img_id)
            sample["ground_truth"] = fo.Detections(
                detections = [
                    fo.Detection(
                        bounding_box = bbox
                    )
                    for bbox in bbox_list(img_id)
                ]
            )
            dataset1.add_sample(sample)
            if idx> 1000:
                break

    # dataset = fo.load_dataset(f"{dataset_type}_dataset")
    dataset = fo.load_dataset(f"All_dataset")
    
    session = fo.launch_app(dataset)
    session.wait()
    print(dataset1)
