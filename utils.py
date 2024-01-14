import numpy as np
import tensorflow as tf

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    intersection = tf.minimum(boxes1[..., 0], boxes2[..., 0]) * tf.minimum(boxes1[..., 1], boxes2[..., 1])
    union = (boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection)
    return intersection / union


def intersection_over_union(true_box, pred_box, box_format="corner"):
    true_box = np.array(true_box)
    pred_box = np.array(pred_box)

    if box_format=="midpoint":
        pass
    if box_format=="corner":
        box1_x1 = true_box[...,0:1]
        box1_y1 = true_box[...,1:2] 
        box1_x2 = true_box[...,2:3]
        box1_y2 = true_box[...,3:4]
        
        box2_x1 = pred_box[...,0:1]
        box2_y1 = pred_box[...,1:2]
        box2_x2 = pred_box[...,2:3]
        box2_y2 = pred_box[...,3:4]

    I_x1 = np.maximum(box1_x1,box2_x1)
    I_x2 = np.minimum(box1_x2,box2_x2)
    
    I_y1 = np.maximum(box1_y1,box2_y1)
    I_y2 = np.minimum(box1_y2,box2_y2)
    
    intersection = (I_x2-I_x1).clip(0)*(I_y2-I_y1).clip(0)
    
    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    
    return intersection /(box1_area+box2_area-intersection+1e-6)


def non_max_suppression(predictions, iou_threshold, prob_threshold, box_format = 'corners'):
    '''prediction = [class, prob, x1, y1, x2, y2]'''
    assert type(boxes) == list
    
    bboxes_after_nms = []
    bboxes = [box for box in boxes if box[1] > threshold]
    bboxes = sorted(bboxes,key = lambda x:x[1], reverse = True)

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(tf.Tensor(chosen_box[2:]), tf.Tensor(box[2:]), box_format='corners')]
        bboxes_after_nms.append(bboxes)
    return bboxes_after_nms

import tensorflow as tf

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they, for example, can be plotted.
    
    INPUT:
    predictions: tensor of size (BATCH_SIZE, S, S, num_anchors, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided into on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (BATCH_SIZE, num_anchors * S * S, 6) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = tf.shape(predictions)[0]
    num_anchors = len(anchors)
    
    box_predictions = predictions[..., 1:5]
    
    if is_preds:
        anchors = tf.reshape(anchors, (1, 1, 1, num_anchors, 2))
        box_predictions[..., 0:2] = tf.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = tf.exp(box_predictions[..., 2:]) * anchors
        scores = tf.sigmoid(predictions[..., 0:1])
        best_class = tf.argmax(predictions[..., 5:], axis=-1, output_type=tf.dtypes.int32)
        best_class = tf.expand_dims(best_class, axis=-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = tf.reshape(tf.range(S), (1, 1, S, 1))
    cell_indices = tf.tile(cell_indices, (BATCH_SIZE, num_anchors, 1, S))

    x = 1 / S * (box_predictions[..., 0:1] + tf.cast(cell_indices, dtype=tf.float32))
    y = 1 / S * (box_predictions[..., 1:2] + tf.transpose(tf.cast(cell_indices, dtype=tf.float32), perm=(0, 1, 3, 2)))
    w_h = 1 / S * box_predictions[..., 2:4]

    converted_bboxes = tf.concat([tf.cast(best_class, dtype=tf.float32), scores, x, y, w_h], axis=-1)
    converted_bboxes = tf.reshape(converted_bboxes, (BATCH_SIZE, num_anchors * S * S, 6))

def load_checkpoint():
    pass