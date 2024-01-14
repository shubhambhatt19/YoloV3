import tensorflow as tf
import numpy as np
from utils import intersection_over_union

# s = split size
# b = boxes
# c = class

def YoloLoss(predictions, target, anchors, batch_size, output_size, NUM_CLASS):

    lambda_class = 1
    lambda_noobj = 10
    lambda_obj = 1
    lambda_box = 10
    

    # This is the shape of the bounding box coordinates and we have set the maximum detection as 150.
    # target[1].shape = (1, 150, 4)

    # This is the shape of the sub-image with 32*32*3 size with 20 class +  
    # target[0].shape = (1, 32, 32, 3, 25)

    obj = target[..., 0] == 1  # in paper this is Iobj_i
    noobj = target[..., 0] == 0  # in paper this is Inoobj_i

    # predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1], predictions.shape[2], 3, 5 + NUM_CLASS))
    # target = target[0]

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    cce = tf.keras.losses.CategoricalCrossentropy()
    sigmoid = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    # ======================= #
    #   FOR NO OBJECT LOSS    #
    # ======================= #
    no_object_loss = bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))

    # ==================== #
    #   FOR OBJECT LOSS    #
    # ==================== #

    box_preds = tf.concat([tf.math.sigmoid(predictions[..., 1:3]), tf.math.exp(predictions[..., 3:5])], axis=-1) # * anchors
    ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj])#.detach()
    object_loss = mse(tf.math.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

    # ======================== #
    #   FOR BOX COORDINATES    #
    # ======================== #
    pred = predictions.numpy()
    pred[..., 1:3] = 1 / (1 + np.exp(-pred[..., 1:3]))  # sigmoid
    target[..., 3:5] = np.log((1e-16 + target[..., 3:5])) # / anchors # width, height coordinates
    box_loss = mse(pred[..., 1:5][obj], target[..., 1:5][obj])

    # ================== #
    #   FOR CLASS LOSS   #
    # ================== #
    class_loss = cce((pred[..., 5][obj]), (target[..., 5][obj]),)
    loss = (lambda_box*box_loss) + (lambda_obj*object_loss) + (lambda_noobj*no_object_loss) + (lambda_class * class_loss)
    return loss