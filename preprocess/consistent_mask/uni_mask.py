# convert RGB mask to a gray mask

import os
import numpy as np
from PIL import Image

# get mask_root from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mask_root", type=str, required=True) # example: /home/zhangyupeng/w/3drecon/LangSplat/dataset/3d_ovs/sofa/mask/mask_x/
args = parser.parse_args()
mask_root = args.mask_root

global_ids = {}

def rgb2instanceID(rgb):
    return np.dot(rgb[...,:3], [1, 256, 65536]).astype(np.uint32)

def convert_mask():
    anno_dir = mask_root + "/Annotations"
    mask_files = os.listdir(anno_dir)
    #print("mask_files:", mask_files)

    #create a new dir to save the mask file
    mask_dir = os.path.join(mask_root, "uni_mask")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for mask_file in mask_files:
        # get base name of mask_file
        mask_file = os.path.basename(mask_file).split(".")[0]

        mask_path = os.path.join(anno_dir, mask_file + ".png")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        instance_id = rgb2instanceID(mask)
        ids = np.unique(instance_id)

        # for each id in ids, add to global_id if not exist
        for id in ids:
            if id not in global_ids:
                global_ids[id] = len(global_ids)
        
        # convert instance_id to global_id
        for id in global_ids:
            instance_id[instance_id == id] = global_ids[id]
        instance_id = instance_id.astype(np.int16)
        np.save(os.path.join(mask_dir, mask_file + ".npy"), instance_id)

    print("global_ids:", global_ids, len(global_ids))
    print("save to", os.path.join(mask_dir))
    # save global_ids to a file
    with open(os.path.join(mask_root, "global_ids.txt"), "w") as f:
        for id in global_ids:
            f.write(str(global_ids[id]) + "\n")
        
    print(os.path.join(mask_root, "global_ids.txt"), len(global_ids))

convert_mask()
