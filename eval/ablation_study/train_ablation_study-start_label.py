import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

start_cmd = ""

iteration = 15000
start_version = 43
mask_version = 3


start_label_iters = [1]
#for i in range(1):
#    start_label_iters.append(i*1000)

#print(start_label_iters)
#exit()

scene_name = "bed"
for idx, start_label in enumerate(start_label_iters):

    version = start_version + idx

    cmd = (f"python train.py -s dataset/{dataset_name}/{scene_name} -m output/{dataset_name}/auto_{scene_name}_segEval{version} --eval --iteration {iteration} --label {start_cmd} --occlude_flag --mask_version {mask_version} --gpf_flag --start_label_iter {start_label}") 
    print(cmd)
    os.system(cmd)
    #exit()

