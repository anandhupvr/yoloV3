import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.net import Anet
import config.parameters as p
from loss import loss
import cv2
from utils import datamaker


config = p.getParams()
images, labels = datamaker.get_data(config)
arch = Anet(config)
preds = arch.network()
x = arch.getX()
with tf.Session() as sess:
    loss = loss.yolo_loss(preds, config, labels)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.global_variables_initializer())
    sess.run(train_step, feed_dict={x:images})
    # writer =  tf.summary.FileWriter("/run/media/anandhu/baskaran/agrima/git_repos/yoloV2", sess.graph)