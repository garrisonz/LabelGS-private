import torch
import torchvision
import imageio

# add label to the 3d gaussian accordding to the lable on 2D image after mask.
def lift_label(gaussian_labels, alpha_max_map, orig_alpha_id_map, mask_map, proj_means2D, label_id, opt, iteration=0, out_dir="",alpha_max_threshold=0, gpf_flag=True):
    """
    Args:
        gaussian_labels: torch.Tensor, shape=(N, 3), dtype=torch.float32
        alpha_max_map: torch.Tensor, shape=(H, W), dtype=torch.float32
        alpha_id_map: torch.Tensor, shape=(H, W), dtype=torch.int32
        mask_map: torch.Tensor, shape=(H, W), dtype=torch.int32
        proj_means2D: torch.Tensor, shape=(N, 2), dtype=torch.long the projection of all gaussians on image space
        label_id: int
    """
    H, W = mask_map.shape

    alpha_id_map = orig_alpha_id_map + 1

    # clean the alpha_id_map by mask_map
    alpha_id_map = alpha_id_map * mask_map
    alpha_id_map -=1

    alpha_id_map[alpha_max_map < alpha_max_threshold] = -1

    train_path = out_dir + "/train"

    # get all non-zero elements in the alpha_id_map
    alpha_ids = alpha_id_map[alpha_id_map >= 0]

    # print unique value of alpha_id_map
    opacity_stable = False
    if iteration % opt.opacity_reset_interval > 100:
        opacity_stable = True
    if opacity_stable and alpha_ids.shape[0] == 0:
        #print("[warning] no alpha_ids to lift label. label_id:", label_id)
        return 0

    # get the unique alpha_id in the alpha_id_map
    alpha_ids = torch.unique(alpha_ids).type(torch.long)
    update_gid = alpha_ids


    # start get update_gid by mask
    visible_gid = torch.where((proj_means2D[:, 0] >= 0) & (proj_means2D[:, 0] < W) &
                                (proj_means2D[:, 1] >= 0) & (proj_means2D[:, 1] < H))[0]
    
    # start. Gaussian Projection Filter
    if gpf_flag:
        indicate = mask_map[proj_means2D[visible_gid][:, 1], proj_means2D[visible_gid][:, 0]]
        valid_gid = visible_gid[indicate]
    else:
        valid_gid = visible_gid
        print("[Warning]not use GPF", gpf_flag)
    # end. Gaussian Projection Filter

    update_gid = alpha_ids[torch.isin(alpha_ids, valid_gid)].type(torch.long)
    # end get update_gid by mask

    if (update_gid > gaussian_labels.shape[0]).any():
        print("error: alpha_ids > gaussian_labels.shape[0]")
        print(update_gid.shape)
        print(update_gid)
        print(gaussian_labels.shape)
        exit()

    # assigned the label_id to the gaussian_labels
    gaussian_labels[update_gid] = label_id
    return update_gid.shape[0]


def test_lift_label():
    # test lift_label
    gaussian_labels = torch.tensor([-1, -1, -1, -1, 0, 0])
    # create alpha_id_map in shape (H, W)
    alpha_id_map = torch.tensor([[0, 1, 1, -1, 0],
                                 [4, 4, 0, 3, 0],
                                 [0, 0, 2, 0, 0],
                                 [0, 0, -1, -1, -1],
                                 [5, -1, -1, -1, -1]], dtype=torch.int32).unsqueeze(0)
    alpha_max_map = torch.tensor([[0, 1, 0.9, 0, 0],
                                  [0.9, 0.5, 0, 0.8, 0],
                                  [0, 0, 0.9, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0.2, 0, 0, 0]], dtype=torch.float32).unsqueeze(0)
    mask_map = torch.tensor([[1, 1, 1, 1, 0],
                             [1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=torch.int32).unsqueeze(0)
    lift_label(gaussian_labels, alpha_max_map, alpha_id_map, mask_map, 0)



if __name__ == "__main__":
    test_lift_label()