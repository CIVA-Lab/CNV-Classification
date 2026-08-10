"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

from matplotlib import pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from PIL import Image
import scipy.misc
import imageio.v2 as imageio

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs-cornea-multi")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/CNV_NEW_MULTI/")

# VAL_IMAGE_IDS = ["image2"]
VAL_IMAGE_IDS = ['177_HM_Day 0_177_HM_Day 0_Image4', 
              "Animal 9_Image1",
              'Day 21_ET091_04.23.2015_2.5X_Image1',
              "Day 28_ET092_04.29.2015_2.5X_Image1",
              "ET_115_Day 14_04.09.2015_Image1",
              "ROCK_WT_159_Day 35_Image1"]

VAL_Grades = ["grade0",
                 'grade0',
                 'grade1',
                 "grade2",
                 "grade3",
                 "grade4"]

TRAIN_IMAGE_IDS = ['129_WT_Day 0_129_WT_Day 0_Image1',
                   '131_HT_Day 0_131_HT_Day 0_Image1', 
                   'Animal 2_Image1',
                   'Animal 3_Image1',
                   'Animal 5_Image1',
                   'Animal 7_Image1',
                   'Day 15_ET043_04.17.2015_2.5X_Image1',
                   'Day 21_ET076_04.23.2015_2.5X_Image1',
                   'Day 28_ET043_04.29.2015_2.5X_Image1',
                   'ET_043_Day 37_05.07.2015_Image1',
                   'ET_078_Day 37_05.07.2015_Image1',
                   'ET_092_Day 37_05.07.2015_Image1',
                   'ET_101_Day 0_03.26.2015_Image1',
                   'ET_101_Day 14_04.09.2015_Image1',
                   'ET_101_Day 21_04.16.2015_Image1',
                   'ET_102_Day 14_04.09.2015_Image1',
                   'ET_103_Day 40_05.07.2015_Image1',
                   'ET_103_Day 7_04.02.2015_Image1',
                   'ET_110_Day21_04.16.2015_Image1',
                   'HM_162_Day 28_Image1',
                   'HT006_d-21_04.23.2015_2.5& 3.2 X_Image1',
                   'HT006_d 37_05.07.2015_2. X_Image1',
                   'HT_053_Day 28_Image1',
                   'HT_053_Day 35_Image1',
                   "HT_059_Day 35_Image1",
                   "HT049_d-28_04.29.2015_2. X_Image1"]

TRAIN_Grades = ['grade0',
                'grade0',  
                'grade0',
                'grade0',
                'grade0',
                'grade0',
                'grade1',
                'grade1',
                'grade2',
                'grade3',
                'grade1',
                'grade3',
                'grade0',
                'grade3',
                'grade4',
                'grade1',
                'grade2',
                'grade0',
                'grade4',
                'grade0',
                'grade2',
                'grade4',
                'grade1',
                'grade3',
                "grade2",
                "grade4"]
############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "cornea"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("cornea", 1, "grade0")
        self.add_class("cornea", 2, "grade1")
        self.add_class("cornea", 3, "grade2")
        self.add_class("cornea", 4, "grade3")
        self.add_class("cornea", 5, "grade4")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
            image_grades = VAL_Grades
        else:
            # Get image ids from directory names
            # image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = TRAIN_IMAGE_IDS
                image_grades = TRAIN_Grades
            else:
                image_ids = next(os.walk(dataset_dir))[1]
                image_grades = next(os.walk(dataset_dir))[1]

        # Add images
        for i, image_id in enumerate(image_ids):
            self.add_image(
                "cornea",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)), 
                cornea = image_grades[i])
            
            # print(f"Index {i}, Image ID: {image_id}")  # Example of using i
            # print(f"Index {i}, Grade: {image_grades[i]}")  # Example of using i

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # print("Info: ", info)
        cornea = info['cornea']
        # print("Cornea:", cornea)

        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(cornea)])
        # print("Class IDS:", class_ids)
        return mask, class_ids.astype(np.int32)

        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cornea":
            return info["cornea"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores,submit_dir):
    "Encodes instance masks to submission format."
    org_mask = mask
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    # 2 lines added by yasmin
    submit_dir_mask = "{}".format(image_id)
    submit_dir_mask = os.path.join(submit_dir, submit_dir_mask)
    if not os.path.exists(submit_dir_mask):
        os.makedirs(submit_dir_mask)
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
	# 1 line added by yasmin
        print("{}/{}_{}.png".format(submit_dir_mask,image_id,o))
        #scipy.misc.imsave("{}/{}_{}.png".format(submit_dir_mask,image_id,o), m)
        imageio.imwrite("{}/{}_{}.png".format(submit_dir_mask,image_id,o), org_mask[:, :, o-1].astype("uint8") * 255)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        # rle = mask_to_rle(source_id, r["masks"], r["scores"],submit_dir)
        # submission.append(rle)

        ###### yasmin added 6 lines to store rois
        submit_dir_mask = "{}".format(source_id)
        submit_dir_mask = os.path.join(submit_dir, submit_dir_mask)
        if not os.path.exists(submit_dir_mask):
            os.makedirs(submit_dir_mask)
        for i in range(r["masks"].shape[2]):
            print("{}/{}_{}.png".format(submit_dir_mask,source_id,i+1))
            #scipy.misc.imsave("{}/{}_{}.png".format(submit_dir_mask,image_id,o), m)
            imageio.imwrite("{}/{}_{}.png".format(submit_dir_mask,source_id,i+1), r["masks"][:, :, i].astype("uint8") * 255)

        matrois = np.matrix(r['rois'])
        matid = np.matrix(r['class_ids'])
        matscores = np.matrix(r['scores'])
        np.savetxt('{}/{}.mat'.format(submit_dir,source_id),matrois)
        np.savetxt('{}/{}_id.mat'.format(submit_dir,source_id),matid)
        np.savetxt('{}/{}_scores.mat'.format(submit_dir,source_id),matscores)

        print("Image: ", source_id)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions", ax=ax  # Pass axis explicitly
        )
        
        # Force rendering before saving
        plt.draw()

         # Save the figure
        plt.savefig(f"{submit_dir}/{source_id}.png", bbox_inches='tight', dpi=300)

        # Show figure (for debugging, optional)
        plt.show()

        # Close figure to free memory
        plt.close(fig)
        
    # # Save to csv file
    # submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    # file_path = os.path.join(submit_dir, "submit.csv")
    # with open(file_path, "w") as f:
    #     f.write(submission)
    # print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
