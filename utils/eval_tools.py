import os
import numpy as np

# views is list of Camera
def interpolate_views(views, num_interpolations=10):
    new_views = []
    for i in range(len(views) - 1):
        view1 = views[i]
        view2 = views[i + 1]

        T1 = view1.T # shape [3]
        T2 = view2.T # shape [3]
        R1 = view1.R # shape [4]
        R2 = view2.R # shape [4]

        new_views.append(view1)
        for i in range(1, num_interpolations):
            T = T1 + (T2 - T1) * i / num_interpolations

            # R is a 3x3 rotation matrix, so we need to interpolate it by slerp
            from scipy.spatial.transform import Rotation as R

            # R1 and R2 are the input rotation matrices
            rotation1 = R.from_matrix(R1)
            rotation2 = R.from_matrix(R2)

            # Convert to quaternions
            q1 = rotation1.as_quat()
            q2 = rotation2.as_quat()

            import numpy as np

            def slerp(q1, q2, t):
                dot_product = np.dot(q1, q2)
                if dot_product < 0.0:
                    q2 = -q2
                    dot_product = -dot_product

                dot_product = np.clip(dot_product, -1.0, 1.0)
                theta_0 = np.arccos(dot_product)
                theta = theta_0 * t

                q3 = q2 - q1 * dot_product
                q3 = q3 / np.linalg.norm(q3)

                return q1 * np.cos(theta) + q3 * np.sin(theta)

            # Perform SLERP interpolation for a parameter t
            qt = slerp(q1, q2, i/num_interpolations)
            #print(qt)
            rotation_t = R.from_quat(qt)
            R_t = rotation_t.as_matrix()
            #print(R_t.shape)

            # import PseudoCamera from scene/cameras.py
            from scene.cameras import PseudoCamera

            new_views.append(PseudoCamera(
                R=R_t, T=T, FoVx=view1.FoVx, FoVy=view1.FoVy,
                width=view1.image_width, height=view1.image_height
            ))
        #print("1r new_views:", len(new_views))
    new_views.append(views[-1]) 
    return new_views


# frame_name:
# |--- object_name1 : seg1
# |--- object_name2 : seg2
#
def get_prompt_and_eval(seg_path):
    # each file in seg_path is a segmentation mask, with label on file name

    anno_all = {}
    frame_list = []
    for seg_frame in os.listdir(seg_path):
        if not os.path.isdir(os.path.join(seg_path, seg_frame)):
            continue
        frame_list.append(seg_frame)
    frame_list.sort()

    for seg_frame in frame_list:
        sef_frame_dir = os.path.join(seg_path, seg_frame)

        anno = {}
        for seg_name in os.listdir(sef_frame_dir):
            seg_file = os.path.join(sef_frame_dir, seg_name)
            label = seg_name.split(".")[0]
            #print(f"{seg_file}")

            # read the segmentation mask, paired with label, and stored in anno
            import cv2
            seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
            seg = seg > 128
            anno[label] = seg
        anno_all[seg_frame] = anno
    print(anno_all.keys())
    prompt = {}
    first = list(anno_all.keys())[0]
    prompt[first]= anno_all[first]
    eval_set = {k: v for k, v in anno_all.items() if k != first}
    return prompt, eval_set

def get_bbox_from_seg2(seg, bbox_padding_side_ratio = 0.15):
    # seg is a numpy array
    # find the bbox contains all true in seg map and keep the ratio as seg shape
    height, width = seg.shape
    seg_ind = np.where(seg)
    min_x, min_y = seg_ind[0].min(), seg_ind[1].min()
    max_x, max_y = seg_ind[0].max(), seg_ind[1].max()
    half_x = (max_x - min_x) * bbox_padding_side_ratio
    half_y = (max_y - min_y) * bbox_padding_side_ratio
    min_x = max(min_x - half_x, 0)
    min_y = max(min_y - half_y, 0)
    max_x = min(max_x + half_x, height - 1)
    max_y = min(max_y + half_y, width - 1)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    crop_heigh = max_x - min_x
    crop_width = max_y - min_y
    if crop_heigh / height >= crop_width / width:
        ratio = crop_heigh / height
    else:
        ratio = crop_width / width
    #print(ratio)
    bbox_width = width * ratio
    bbox_heigh = height * ratio

    min_x = center_x - bbox_heigh / 2
    min_y = center_y - bbox_width / 2
    max_x = center_x + bbox_heigh / 2
    max_y = center_y + bbox_width / 2

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, height - 1)
    max_y = min(max_y, width - 1)
    
    return min_x, min_y, max_x, max_y