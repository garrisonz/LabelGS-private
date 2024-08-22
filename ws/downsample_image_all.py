import os

dataset_path = "/home/zhangyupeng/w/3drecon/LabelGS/dataset/nerf_llff_data"

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
print("scene_names:", scene_names)

#scene_names = ['bench', 'bed', 'snacks', 'table', 'lawn', 'blue_sofa', 'covered_desk', 'office_desk']

scene_names.sort()

for scene_name in scene_names:
    scene_path = f"{dataset_path}/{scene_name}"
    print("downscale: ", scene_path)

    os.system(f"python ws/downsample_image2.py --scene_path {scene_path}") 