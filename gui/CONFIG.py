import os

class CONFIG:
    r = 2 
    window_width = int(1440/r)
    window_height = int(1080/r)

    width = int(1440/r)
    height = int(1080/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    MODEL_PATH = './output/'
    SOURCE_PATH = 'dataset'

    SCENE_GAUSSIAN_ITERATION = 15000

    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')
