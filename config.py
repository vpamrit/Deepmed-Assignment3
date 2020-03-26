#file settings
SAVE_DIR = './saved_models/unet/'
TRAIN_DIR = 'img/'
IMG_PREFIX = 'img'
LABEL_DIR = 'label/'
LABEL_PREFIX = 'label'
EXT = '.nii.gz'


# various
SAVE_VALID_IMAGES = False
SAVE_EPOCHS = [0, 1, 2]
CLASSES = [1,6,7,8,9,11]
WEIGHTS = [1, 3, 1, 1, 1, 1, 1]
SLICE_SIZE = 256

