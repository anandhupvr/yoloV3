import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from network.net import Anet
import config.parameters as p
from loss import yolo_loss
import cv2
from utils.datamaker import DataLoader
from utils import loader
import sys


config = p.getParams()
data_path = sys.argv[1]
# images, labels = datamaker.get_data(config)
data_loader = DataLoader(data_path, config)
arch = Anet(config)

preds = arch.network()
# var_list = [v for v in tf.trainable_variables()]
# path = "/run/media/anandhu/disk2/agrima/git_repos/yoloV2/yolov2-voc_100.weights"
# offset = 4
# all_weights = np.fromfile(path)
# offset = offset + 19
# weights = all_weights[offset - 19:offset]

# arch.weight_load(weights, var_list)

x = arch.getX()
epochs = 10
# for v in tf.trainable_variables():
# 	print ( v)
# 	input()


with tf.Session() as sess:
	for epoch in range(epochs):
		x_batch, y_batch = data_loader.next_batch(10)
		loss = yolo_loss.loss(preds, config, y_batch)
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
		sess.run(tf.global_variables_initializer())
		sess.run(train_step, feed_dict={x:x_batch})
		print("total loss : "+str(tf.Print(loss)) + str(epoch) + ": epochs")