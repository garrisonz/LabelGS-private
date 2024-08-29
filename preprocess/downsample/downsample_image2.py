import os
import cv2
import numpy as np

# get scene_path from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str, required=True) # "dataset/3d_ovs/sofa"
args = parser.parse_args()
scene_path = args.scene_path
print("scene_path:", scene_path)


def downsample_seg():
    origin_path = scene_path + "/segmentations"
    bak_path = scene_path + "/segmentations_input"

    if not os.path.exists(bak_path):
        os.system(f"mv {origin_path} {bak_path}")
        print(f"mv {origin_path} {bak_path}")

    os.makedirs(origin_path, exist_ok=True)

    for frame in os.listdir(bak_path):
        if not os.path.isdir(os.path.join(bak_path, frame)):
            continue
        downsample_image(os.path.join(bak_path, frame), os.path.join(origin_path, frame), is_mask=True) 

#downsample_seg()

# downsample images in img_dir to img_dir_8 with 8x downsample
def downsample_image(source_folder, output_folder, is_mask=False):
    img_dir = source_folder
    output_folder = output_folder
    os.makedirs(output_folder, exist_ok=True)

    #remove img_dir_8
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(img_dir)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        if not img_name.endswith(".jpg") and not img_name.endswith(".png") and not img_name.endswith(".JPG"):
            continue
        img = cv2.imread(img_path)

        image = img
        orig_w, orig_h = image.shape[1], image.shape[0]
        if orig_h > 1080:
            global_down = orig_h / 1080
        else:
            global_down = 1
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)
        img = image

        if is_mask:
            img = img > 128
            img = img.astype(np.uint8) * 255

        cv2.imwrite(os.path.join(output_folder, img_name), img)
        print(f"downsampled {img_name} to {output_folder} {img.shape}")


origin_path = scene_path + "/images"
bak_path = scene_path + "/images_input"
if not os.path.exists(bak_path):
    os.system(f"mv {origin_path} {bak_path}")
    print(f"mv {origin_path} {bak_path}")

downsample_image(scene_path + "/images_input", scene_path + "/images")
