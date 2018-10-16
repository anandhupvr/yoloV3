import tensorflow as tf
import numpy as np


def loss(y_pred, config, y_true):
    # pred_xy = np.reshape(config["ANCHORS"], [1, 1, 1, config["BOX"], 2])
    # p_xy = np.reshape(config["ANCHORS"], [1, 1, 1, config["BOX"], 2])
    mask_shape = tf.shape(y_true)[:4]
    conf_mask  = tf.zeros(mask_shape)
    true_xy = (y_true[..., :2])
    pred_xy = tf.sigmoid(y_pred[..., :2])

    true_wh = (y_true[..., 2:4])
    pred_wh = tf.exp(y_pred[..., 2:4])

    pred_classes = y_pred[..., 5:]

    pred_objectness = y_pred[..., 4]
    pred_class = y_pred[..., 5:]
    true_classes = tf.argmax(y_true[..., 5:], -1)
    # compute iou

    pred_wh_half = pred_wh / 2
    pred_min = pred_xy - pred_wh_half
    pred_max = pred_xy + pred_wh_half
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]

    true_wh_half = true_wh / 2
    true_min = true_xy - true_wh_half
    true_max = true_xy + true_wh_half
    true_area = true_wh[..., 0] * true_wh[..., 1]

    intersection_cords_min = tf.maximum(pred_min, true_min)
    intersection_cords_max = tf.minimum(pred_max, true_max)
    intersection_wh = tf.maximum(intersection_cords_max - intersection_cords_min, 0)
    intersection_area = intersection_wh[..., 0] * intersection_wh[...,1]

    total_area = true_area + pred_area - intersection_area

    iou_scores = intersection_area / total_area

    true_box_conf = iou_scores * y_true[..., 4]
    pred_box_conf = y_pred[..., 4]
    class_wt = np.ones(config["CLASS"])

    conf_mask = conf_mask + tf.to_float(iou_scores < 0.6) * (1 - y_true[..., 4]) * config["NO_OBJECT_SCALE"]
    conf_mask = conf_mask + y_true[..., 4] * config["OBJECT_SCALE"]
    class_mask = y_true[..., 4] * tf.gather(class_wt, true_classes) * config["CLASS_SCALE"]
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * config["COORD_SCALE"]
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    class_mask = tf.to_float(class_mask)

    loss_xy = tf.reduce_sum(tf.square(pred_xy - true_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(pred_wh - true_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2. 
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=pred_classes)
    loss_class = tf.reduce_sum(loss_class * class_mask)  / (nb_class_box + 1e-6)
    
    return (loss_xy + loss_wh + loss_conf + loss_class)
    


