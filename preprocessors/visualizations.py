import matplotlib.pyplot as plt
import scipy.io
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT_FOLDER_NYU = '/Volumes/Evan_Samsung_HP_data/nyu_dataset/data/'
ROOT_FOLDER_BIG22M = '/Volumes/Evan_Samsung_HP_data/'
TEST_FOLDER = ROOT_FOLDER_NYU + 'test/'
TRAIN_FOLDER = ROOT_FOLDER_NYU + 'train/'
# JOINT_LIST_NYU = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
# JOINT_LIST_NYU = [0]
# JOINT_LIST_NYU_MATCH_BIG22 = [31, 5, 4, 2, 0, 11, 10, 8, 6, 17, 16, 14, 12, 23, 22, 20, 18, 29, 28, 26, 24]
# JOINT_LIST_NYU_MATCH_BIG22 = [31, 28, 27, 25, 24] # thumb
# JOINT_LIST_NYU_MATCH_BIG22 = [31, 23, 22, 20, 18] # index finger
# JOINT_LIST_NYU_MATCH_BIG22 = [31, 17, 16, 14, 12,] # Middle finger
# JOINT_LIST_NYU_MATCH_BIG22 = [31, 11, 10, 8, 6] # ring finger
# JOINT_LIST_NYU_MATCH_BIG22  = [31, 5, 4, 2, 0] # pinky
JOINT_LIST_NYU_MATCH_BIG22 = [31, 28, 23, 17, 11, 5, 27, 22, 16, 10, 4, 25, 20, 14, 8, 2, 24, 18, 12, 6, 0]
JOINT_LIST_BIG22M = [0]
N = 1003


def plot_keypoints_NYU(mat_file, image_file):
    # Load keypoints from .mat file
    data = scipy.io.loadmat(mat_file)
    # This assumes 'keypoints' is the name of the variable in your .mat file
    # print(data)
    keypoints = data['joint_uvd']
    # hand = keypoints[0][0][JOINT_LIST_NYU] # the 1rst camera, the 1rst example
    # joint_names = data['joint_names'][0][JOINT_LIST_NYU]
    # hand = keypoints[0][0] # use all the joints
    # joint_names = data['joint_names'][0] # use all the joints
    hand = keypoints[0][N - 1][JOINT_LIST_NYU_MATCH_BIG22]
    joint_names = data['joint_names'][0][JOINT_LIST_NYU_MATCH_BIG22]
    joint_names_print = data['joint_names'][0]

    print("joint names:")

    print(joint_names_print)
    print(f'length of joints: {len(JOINT_LIST_NYU_MATCH_BIG22)}')

    # Load the image
    image = Image.open(image_file)

    # Draw circles at keypoints
    draw = ImageDraw.Draw(image)
    try:
        fnt = ImageFont.truetype("OpenSans_Condensed-Medium.ttf", 8)
    except IOError:
        print("Didn't work")
        fnt = ImageFont.load_default()

    fnt = ImageFont.load_default()
    i = 0
    for point, label in zip(hand, joint_names):
        label = str(label[0])
        print(f'{label}: {JOINT_LIST_NYU_MATCH_BIG22[i]}')
        i += 1
        x, y, z = point
        # Draw a circle at the keypoint
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline='red', width=2)
        draw.ellipse((10, 10, 20, 20), outline='blue', width=2)
        draw.ellipse((40, 10, 50, 20), outline='green', width=2)
        draw.ellipse((x + 20 - 5, y - 5, x + 20 + 5, y + 5), outline='purple', width=2)

        # Draw the joint name. Adjust positioning as needed.
        draw.text((x, y), label, fill=(255, 0, 255), font=fnt)

        # Display the image
    original_img = Image.open(image_file)
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title("Joints")
    axs[1].imshow(original_img)
    axs[1].axis('off')
    axs[1].set_title("Original")
    plt.show()


def plot_keypoints_BIG22M(keypoints, image_file):
    pass


# Example usage
# plot_keypoints('path_to_mat_file.mat', 'path_to_image.jpg')
if __name__ == "__main__":
    test_mat = TEST_FOLDER + 'joint_data.mat'

    n = str(N)
    imn = 'rgb_1_' + n.zfill(7) + '.png'
    img_name = TEST_FOLDER + imn
    plot_keypoints_NYU(test_mat, img_name)