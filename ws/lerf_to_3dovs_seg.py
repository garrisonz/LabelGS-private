# convert segmentation mask from lerf_ovs dataset format to 3dovs dataset format
# zyp 
import os
import numpy as np
import cv2

def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask

def convert():
    lerf_ovs_path = "dataset/lerf_ovs"
    label_path = lerf_ovs_path + "/label"

    scenes = os.listdir(label_path)
    scenes = [scene for scene in scenes if scene[0]!="."]
    scenes.sort()
    for scene in scenes:
        print(scene)
        scene_label = os.path.join(label_path, scene)
        print(scene_label)

        anno_files = os.listdir(scene_label)
        frames = [anno_file.split(".")[0] for anno_file in anno_files]
        frames = [frame for frame in frames if frame!=""]
        frames = list(set(frames))
        frames.sort()
        print(frames)

        for frame in frames:
            json_file = os.path.join(scene_label, frame+".json")
            # read json_file
            import json
            with open(json_file, "r") as f:
                data = json.load(f)
            #print(data.keys())
            info = data["info"]
            objects = data["objects"]
            h = info["height"]
            w = info["width"]
            for obj in objects:
                #print(obj)

                label = obj["category"]
                mask = polygon_to_mask((h, w), obj['segmentation'])
                #print(mask)
                # save mask as image file
                mask_save = mask.copy()
                mask_save[mask == 1] = 255
                save_path = os.path.join(lerf_ovs_path, scene, "segmentations", frame, label+".png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, mask_save)
                print(f"save mask to {save_path}")
        print("")

convert()

