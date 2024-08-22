import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--start_hour", type=int, default=-1) # example 3d_ovs

args = parser.parse_args()
dataset_name = args.dataset_name
start_hour = args.start_hour

if start_hour != -1:
    print(start_hour)

# get current time and get the hour number
import datetime
now = datetime.datetime.now()
hour = now.hour

# sleep until 20Aug 03am
if start_hour != -1:
    print(f"sleep until {start_hour}am")
    while hour != start_hour:
        now = datetime.datetime.now()
        hour = now.hour
        print(hour)
        print(f"sleeping... {hour} != {start_hour}")
        os.system("sleep 1h")
    print("wake up!")
    #exit()

#exit()

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

start_cmd = ""

iteration = 15000
mask_version = 3

start_version = 23

alpha_max_thresholds = []
for i in range(10):
    if i <=0:
        continue
    alpha_max_thresholds.append(i * 0.1)

print(alpha_max_thresholds)
#exit()

scene_name = "bed"
for idx, alpha_max_threshold in enumerate(alpha_max_thresholds):

    version = start_version + idx

    if version != 30:
        continue
    start_iteration = 10000
    start_cmd = f"--start_checkpoint output/{dataset_name}/auto_{scene_name}_segEval{version}/chkpnt{start_iteration}.pth"

    cmd = (f"python train.py -s dataset/{dataset_name}/{scene_name} -m output/{dataset_name}/auto_{scene_name}_segEval{version} --eval --iteration {iteration} --label {start_cmd} --occlude_flag --mask_version {mask_version} --gpf_flag --alpha_max_threshold {alpha_max_threshold}") 
    print(cmd)
    os.system(cmd)
    #exit()

