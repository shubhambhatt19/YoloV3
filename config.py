import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import *

DATASET = '/home/bhatt/shubham/Object_detection_implementation/new_implementation/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = False
NUM_WORKERS = 1
BATCH_SIZE = 2
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.1
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "train.csv"#"val2017.txt"
XYSCALE=[1.2, 1.1, 1.05]
STRIDES= [8, 16, 32]
OUTPUT_SIZE = 256

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 
ANCHORS_PER_SCALE = 3

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

models= {"convolutional":{
      "batch_normalize": 1,
      "filters": 64,
      "size": 7,
      "stride": 2,
      "pad": "same",
      "activation": "LeakyReLU",
      "maxpool":"maxpool",
      "size": 2,
      "stride": 2},

"convolutional":{
      "batch_normalize": 1,
      "filters": 192,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU",
      "maxpool":"maxpool",
      "size": 2,
      "stride": 2},
"convolutional":{
      "batch_normalize": 1,
      "filters": 128,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 256,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 256,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU",
      "maxpool":"maxpool",
      "size": 2,
      "stride": 2},
"convolutional":{
      "batch_normalize": 1,
      "filters": 256,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 256,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 256,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 256,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 1024,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU",
      "maxpool":"maxpool",
      "size": 2,
      "stride": 2},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      'batch_normalize': 1,
      "filters": 1024,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "filters": 512,
      "size": 1,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      'batch_normalize': 1,
      "filters": 1024,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "filters": 1024,
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "size": 3,
      "stride": 2,
      "pad": "same",
      "filters": 1024,
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "filters": 1024,
      "activation": "LeakyReLU"},
"convolutional":{
      "batch_normalize": 1,
      "size": 3,
      "stride": 1,
      "pad": "same",
      "filters": 1024,
      "activation": "LeakyReLU"}}



# 9.17AM