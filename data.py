import cv2
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow
from PIL import Image
import config
from utils import iou_width_height as iou
from IPython import embed
import config as cfg
from utils import intersection_over_union , non_max_suppression,load_checkpoint, cells_to_bboxes

class YOLODataset(tf.keras.utils.Sequence):
	def __init__(self, csv_file, img_dir, label_dir, anchors, S=[13, 26, 52], C=20, transform=False):
		self.annotations = pd.read_csv(csv_file)
		self.img_dir = img_dir
		self.label_dir = label_dir
		self.transform = transform
		self.S = S
		self.anchors = tf.convert_to_tensor(anchors[0] + anchors[1] + anchors[2])
		self.num_anchors = 9 # check this harcoded value 9 #self.anchors.shape[0]
		self.num_anchors_per_scale = 3#self.num_anchors //3
		self.C = C
		self.ignore_iou_tresh = 0.5
		self.batch_count = 0
		self.batch_size = cfg.BATCH_SIZE
		self.num_samples = len(self.annotations)
		self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
		
	def __iter__(self):
		return self
	
	def __len__(self):
		return len(self.annotations)

	def __next__(self):
		if self.batch_count > self.num_batchs:
			raise StopIteration
		num = 0
		batch_image = np.zeros((self.batch_size, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))		
		batch_targets = [np.zeros((self.batch_size,self.num_anchors//3, int(S), int(S),6)) for S in self.S] # [prob(obj), x, y, w, h, class]

		while num < self.batch_size:
			index = self.batch_count * self.batch_size + num
			annotation = self.annotations.iloc[index]
			label_path = os.path.join(self.label_dir , self.annotations.iloc[index, 1])
			bboxes = np.roll(np.loadtxt(fname = label_path, delimiter=" ", ndmin=2),4, axis=1).tolist()
			img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
			image = Image.open(img_path).convert("RGB")
			image = image.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
			batch_image[num] = image
			if self.transform:
				augmentations = self.transform(image = image, bboxes=bboxes)
				image = augmentations["image"]
				bboxes = augmentations["bboxes"]
			for box in bboxes:
				iou_anchors = iou(box[2:4], self.anchors)
				anchor_indices = tf.argsort(iou_anchors,direction='DESCENDING', axis=0)# descending = True, 
				x, y, width, height, class_label = box
				has_anchor = [False, False, False]
				for anchor_idx in anchor_indices:
					scale_idx = anchor_idx //self.num_anchors_per_scale # 0,1,2
					anchor_on_scale = anchor_idx % self.num_anchors_per_scale
					S = self.S[scale_idx]
					i, j  = int(S*y), int(S*y)
					anchor_taken = batch_targets[scale_idx][num, anchor_on_scale, i, j, 0]
					if not anchor_taken and not has_anchor[scale_idx]:
						batch_targets[scale_idx][num, anchor_on_scale, i, j, 0] =1
						x_cell = S*x -j 
						y_cell = S*y -i
						width_cell, height_cell = (width*S, height*S) # S =13, width = 0.5, 6.5
						box_cordinates = [x_cell,y_cell,width_cell, height_cell]
						batch_targets[scale_idx][num, anchor_on_scale, i,j,1:5] = box_cordinates
						batch_targets[scale_idx][num, anchor_on_scale, i,j,5] = int(class_label)
						has_anchor[scale_idx] = True
					elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_tresh:
						batch_targets[scale_idx][num, anchor_on_scale, i, j, 0] = -1 # ignore this prediction
					
			num += 1
		self.batch_count += 1
		return [batch_image, batch_targets]


def test():
    anchors = config.ANCHORS
    transform = False
    dataset = YOLODataset("PASCAL_VOC/train.csv","PASCAL_VOC/images/","PASCAL_VOC/labels/",S=[13, 26, 52],anchors=anchors,transform=transform,)
    S = [13, 26, 52]
    scaled_anchors = tf.convert_to_tensor(anchors)
    for x, y in next(dataset):
        boxes = []
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors=anchor)[0]
    

if __name__ == "__main__":
    test()
