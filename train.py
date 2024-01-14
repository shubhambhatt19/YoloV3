
import config as cfg
from IPython import embed
import numpy as np
from tensorflow.keras import Model
from tqdm import tqdm
from model import YOLOv3
from data import YOLODataset
import tensorflow as tf
from utils import intersection_over_union , non_max_suppression,load_checkpoint, cells_to_bboxes
from loss import YoloLoss


DEVICE = 'cuda' if tf.test.is_gpu_available() else 'cpu'
BATCH_SIZE = cfg.BATCH_SIZE
EPOCHS = cfg.EPOCHS
OUTPUT_SIZE = cfg.OUTPUT_SIZE
NUM_CLASS = cfg.NUM_CLASSES
anchors = cfg.ANCHORS
TRANSFORM = cfg.TRANSFORM

			 
def custom_loss(y_true, y_pred):
	losses = (YoloLoss(y_pred, y_true)+YoloLoss(y_pred, y_true)+
			YoloLoss(y_pred, y_true))
	mean_loss = sum(losses)/len(losses) 
	return mean_loss

global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
def train_fn(image, target, model, optimizer, scaled_anchors):
	losses =[]
	y0, y1, y2 = target[0],target[1],target[2]
	with tf.GradientTape() as tape:
		pred_result = model(tf.convert_to_tensor(image), training = True)
		loss_items = YoloLoss(pred_result[0], y0, scaled_anchors[0], batch_size=BATCH_SIZE, output_size=OUTPUT_SIZE, NUM_CLASS=NUM_CLASS)+\
					 YoloLoss(pred_result[1], y1, scaled_anchors[1], batch_size=BATCH_SIZE, output_size=OUTPUT_SIZE, NUM_CLASS=NUM_CLASS)+\
					 YoloLoss(pred_result[2], y2, scaled_anchors[2], batch_size=BATCH_SIZE, output_size=OUTPUT_SIZE, NUM_CLASS=NUM_CLASS)

		losses.append(loss_items)
		grads = tape.gradient(loss_items, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		
		tf.print("=> STEP %4d/%4d lr: %.6f total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),loss_items.numpy()))
		
		
		
def main():	
	global total_steps
	optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
	# input_layer = tf.keras.layers.Input([cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
	model = YOLOv3(input_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), num_classes=cfg.NUM_CLASSES)
	model.summary()
	model.compile(optimizer='adam', loss = custom_loss)
	if cfg.LOAD_MODEL:
		load_checkpoint(cfg.CHECKPOINT_FILE, model, optimizer, cfg.LEARNING_RATE)
	
	anchors = tf.constant(cfg.ANCHORS, dtype=tf.float32)
	anchors = np.array(cfg.ANCHORS)
	S = tf.constant(cfg.S, dtype=tf.float32)
	S = np.array(S)
	S = np.expand_dims(S,1)
	scaled_anchors = anchors * S
	scaled_anchors = tf.convert_to_tensor(scaled_anchors)
	trainset = YOLODataset("PASCAL_VOC/train.csv","PASCAL_VOC/images/","PASCAL_VOC/labels/",S=S,anchors=anchors,transform=TRANSFORM)
	testset = YOLODataset("PASCAL_VOC/test.csv","PASCAL_VOC/images/","PASCAL_VOC/labels/",S=S,anchors=anchors,transform=TRANSFORM)
	total_steps = tf.Variable(tf.math.floordiv(len(trainset),cfg.BATCH_SIZE), trainable=False, dtype=tf.int32)
	for epoch in range(1,cfg.EPOCHS):
		tf.print("### EPOCH %4d" %(epoch))
		for image, target in trainset:
			train_fn(image, target, model, optimizer, scaled_anchors)
			global_steps.assign_add(1)

if __name__ == "__main__":
	main()

