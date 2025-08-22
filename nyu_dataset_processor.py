from scipy.io import loadmat


IMAGE_ROOT = "/Volumes/Evan_Samsung_HP_data/nyu_dataset/data"
IMAGE_TRAIN = IMAGE_ROOT + "/train"
LABEL_TRAIN = IMAGE_ROOT + "/train/joint_data.mat"
LABEL_TEST = IMAGE_ROOT + "/test/joint_data.mat"


def load_nyu_labels():
    # TODO: Currently just loading nyu labels. I want to move this to be part of the data loader class or some function
    # inside the data loader
    # TODO: Make this generic for any filepath
    nyu_train_labels = loadmat(LABEL_TRAIN)
    nyu_test_labels = loadmat(LABEL_TEST)
    nyu_train_labels = nyu_train_labels[]