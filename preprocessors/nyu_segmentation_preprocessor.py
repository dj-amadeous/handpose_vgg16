import cv2
from scipy.io import loadmat
import os, glob
import numpy as np

'''
load the depth images from nyu, find their minimum bounding cube, 
save them segmented onto the hard drive in their new form.
This folder is initially on my external hard drive.
'''

IMAGE_ROOT = "/Volumes/Evan_Samsung_HP_data/nyu_dataset/data"


def bounding_points(key_points):
    x_min, y_min, z_min = np.min(key_points, axis=0)
    x_max, y_max, z_max = np.max(key_points, axis=0)
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ])
    return vertices


def expand_to_cube(vertices):
    """
    the vertices are as defined in the bounding_points function,
     alternating min max on Z, min min max max on Y, and min min min min max max ... on X

    :param vertices: numpy array of the vertices of the bounding box associated with image
    :return: new bounding box that is in the shape of a cube, but still contains the same image.
    """
    x_length = np.abs(vertices[4][0] - vertices[0][0])
    y_length = np.abs(vertices[2][1] - vertices[0][1])
    z_length = np.abs(vertices[1][2] - vertices[0][2])

    max_length = max(x_length, y_length, z_length)

    center = np.mean(vertices, axis=0)

    half_length = (max_length / 2) + 10 # give 10 extra pixels for margin to make sure kp is inside of bounding cube

    cube_vertices = np.array([
        [center[0] - half_length, center[1] - half_length, center[2] - half_length],
        [center[0] - half_length, center[1] - half_length, center[2] + half_length],
        [center[0] - half_length, center[1] + half_length, center[2] - half_length],
        [center[0] - half_length, center[1] + half_length, center[2] + half_length],
        [center[0] + half_length, center[1] - half_length, center[2] - half_length],
        [center[0] + half_length, center[1] - half_length, center[2] + half_length],
        [center[0] + half_length, center[1] + half_length, center[2] - half_length],
        [center[0] + half_length, center[1] + half_length, center[2] + half_length],
    ])

    return cube_vertices


def calculate_min_bound_box(key_points):
    '''
    uvd points are in raw image coordinates, no normalization occurs. the minimum boudning box just barely covers
    the key points. it returns the bounding box that barely fits the key points in the box.
    :param key_points:
    :return: cube vertices, note that they are in x,y,z format
    '''
    com = np.mean(key_points, axis=0)
    bp = bounding_points(key_points)
    cube = expand_to_cube(bp)
    return cube


def process_nyu():
    '''
    TODO: There's a bug in the training cropping when image 8910 has dimensions 233,233 and 234,233
    Since the dimensions don't match, i can't put the cropped image in the new image. I need to
    figure out why they don't match and fix it. Some sort of rounding error when finding the image
    dimensions.
    TODO: I Found the bug. When the bounding box goes off screen python lets you go ahead crop outside of the bounding
    edge. when using numpy it crashes. THe image is only 480x640, but there are pixels at the 481 space of the bounding
    box. I have to cut the corner cases.
    :return:
    '''
    #debug_counter = 0
    test_counter = 0
    train_counter = 0
    debug_value = 5
    test_gt_filepath = IMAGE_ROOT + "/test"
    train_gt_filepath = IMAGE_ROOT + "/train"
    save_path_test = IMAGE_ROOT +"/segmented_test/"
    save_path_train = IMAGE_ROOT + "/segmented_train/"
    test_gt = loadmat(test_gt_filepath + '/joint_data.mat')
    train_gt = loadmat(train_gt_filepath + '/joint_data.mat')
    test_gt = test_gt['joint_uvd']
    train_gt = train_gt['joint_uvd']
    depth_imgs_test = []
    depth_img_test_file_names = glob.glob(test_gt_filepath + "/depth_1_*.png")
    depth_img_test_file_names.sort(key=lambda st: int(st.split('_')[-1].split('.')[0]))
    depth_img_train_filenames = glob.glob(train_gt_filepath + "/depth_1_*.png")
    depth_img_train_filenames.sort(key=lambda st: int(st.split('_')[-1].split('.')[0]))
    for im in depth_img_test_file_names:
        '''if debug_counter > debug_value:
            break'''
        img_ = cv2.imread(im, cv2.IMREAD_ANYDEPTH)
        # Perform processing, binary interpolate to 224x224, 0 pad 2 channels so it's 224x224x3,
        # then save to alternative location on hard drive
        key_points = test_gt[0, test_counter]
        vertices = calculate_min_bound_box(key_points)
        xy_bound = vertices[:, 0:2]
        top_left_x = max(int(np.floor(np.min(xy_bound[:, 0]))), 0)
        top_left_y = max(int(np.floor(np.min(xy_bound[:, 1]))), 0)
        bottom_right_x = min(int(np.round(np.max(xy_bound[:, 0]))), (img_.shape[0] - 1))
        bottom_right_y = min(int(np.round(np.max(xy_bound[:, 1]))), (img_.shape[0] - 1))
        img = np.zeros(((bottom_right_y - top_left_y), (bottom_right_x - top_left_x), 3))
        z_thresh = vertices[:, 2]
        closer_thresh = np.min(z_thresh)
        further_thresh = np.max(z_thresh)
        #cv2.rectangle(img_, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,0,255), thickness=2)
        '''for kp in key_points:
            x = int(np.rint(kp[0]))
            y = int(np.rint(kp[1]))
            cv2.circle(img_, (x,y), 5, (0,0,255), 2)'''

        cropped_img = img_[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        img[:, :, 0] = cropped_img
        img[:, :, 1] = cropped_img
        img[:, :, 2] = cropped_img
        depth_imgs_test.append(cropped_img)
        test_counter += 1
        cv2.imwrite(save_path_test + f"depth_1_{str(test_counter).zfill(7)}_cropped_n_rgb.png", img)
    for im in depth_img_train_filenames:
        '''if debug_counter > debug_value:
            break'''
        img_ = cv2.imread(im, cv2.IMREAD_ANYDEPTH)
        # Perform processing, binary interpolate to 224x224, 0 pad 2 channels so it's 224x224x3,
        # then save to alternative location on hard drive
        key_points = train_gt[0, train_counter]
        vertices = calculate_min_bound_box(key_points)
        xy_bound = vertices[:, 0:2]
        # using ceiling and floor functions so that cropping image keeps same dimensions a few lines down
        top_left_x = max(int(np.floor(np.min(xy_bound[:, 0]))), 0)
        top_left_y = max(int(np.floor(np.min(xy_bound[:, 1]))), 0)
        bottom_right_x = min(int(np.round(np.max(xy_bound[:, 0]))), (img_.shape[0] - 1))
        bottom_right_y = min(int(np.round(np.max(xy_bound[:, 1]))), (img_.shape[0] - 1))
        img = np.zeros(((bottom_right_y - top_left_y), (bottom_right_x - top_left_x), 3))
        z_thresh = vertices[:, 2]
        closer_thresh = np.min(z_thresh)
        further_thresh = np.max(z_thresh)
        #cv2.rectangle(img_, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), thickness=2)
        '''for kp in key_points:
            x = int(np.rint(kp[0]))
            y = int(np.rint(kp[1]))
            cv2.circle(img_, (x,y), 5, (0,0,255), 2)'''

        cropped_img = img_[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        img[:, :, 0] = cropped_img
        img[:, :, 1] = cropped_img
        img[:, :, 2] = cropped_img
        train_counter += 1
        cv2.imwrite(save_path_train + f"depth_1_{str(train_counter).zfill(7)}_cropped_n_rgb.png", img)


def measure_bounding_info_NYU():
    debug_counter = 0
    debug_value = 5
    test_gt_filepath = IMAGE_ROOT + "/test"
    train_gt_filepath = IMAGE_ROOT + "/train"
    test_gt = loadmat(test_gt_filepath + '/joint_data.mat')
    train_gt = loadmat(train_gt_filepath + '/joint_data.mat')
    test_gt = test_gt['joint_uvd']
    train_gt = train_gt['joint_uvd']
    depth_imgs_test = []
    largest_width = 0
    largest_height = 0
    for im in glob.glob(train_gt_filepath + "/depth_1_*.png"):
        #if debug_counter > debug_value:
        #    break
        #img_ = cv2.imread(im, cv2.IMREAD_ANYDEPTH)
        vertices = calculate_min_bound_box(train_gt[0, debug_counter])
        debug_counter += 1
        xmin = vertices[0,0]
        ymin = vertices[0,1]
        xmax = vertices[4,0]
        ymax = vertices[3,1]
        horiz_height = xmax - xmin
        vert_height = ymax - ymin
        if int(horiz_height) > 224 or int(vert_height) > 224:
            print("size exceeded")
            print(f"horizontal size: {horiz_height}")
            print(f"vertical size: {vert_height}")
        if horiz_height > largest_width:
            largest_width = horiz_height
        if vert_height > largest_height:
            largest_height = vert_height
    print("bounds checking complete")
    print(f"largest_width: {largest_width}")
    print(f"largest_height: {largest_height}")



process_nyu()
#measure_bounding_info_NYU()