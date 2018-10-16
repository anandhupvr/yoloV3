import tensorflow as tf
import numpy as np
from utils import util


class Anet:
    def __init__(self, config):
        self.config = config
        self.x = tf.placeholder(
            dtype=tf.float32,
            shape=[
                None,
                config["IMAGE_W"],
                config["IMAGE_H"],
                3])

    def network(self):
        # Layer 1-2
        for i in range(1, 3):
            x_ = tf.layers.conv2d(self.x,
                                  filters=32 * 2,
                                  kernel_size=[3, 3],
                                  strides=(1, 1),
                                  padding="same",
                                  activation=tf.nn.leaky_relu,
                                  name="conv_" + str(i))
            x_ = tf.layers.batch_normalization(
                x_,
                name="norm_" + str(i))
            x_ = tf.nn.max_pool(
                x_,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME")
        # Layer 3
        x_ = tf.layers.conv2d(x_,
                              filters=128,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_3")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_3")
        # Layer 4
        x_ = tf.layers.conv2d(x_,
                              filters=64,
                              kernel_size=[1, 1],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_4")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_4")
        # Layer 5
        x_ = tf.layers.conv2d(x_,
                              filters=128,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_5")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_5")
        x_ = tf.nn.max_pool(x_,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding="SAME")


        # Layer 6
        x_ = tf.layers.conv2d(x_,
                              filters=256,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_6")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_6")
        x_ = tf.nn.max_pool(x_,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding="SAME")


        # Layer 7
        x_ = tf.layers.conv2d(x_,
                              filters=512,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_7")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_7")
        # Layer 8 10
        x_ = tf.layers.conv2d(x_,
                              filters=256,
                              kernel_size=[1, 1],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_8")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_8")
        # Layer 9
        x_ = tf.layers.conv2d(x_,
                              filters=512,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_9")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_9")
        # Layer 10
        x_ = tf.layers.conv2d(x_,
                              filters=256,
                              kernel_size=[1, 1],
                              strides=(1, 1),
                              padding="same",
                              activation=tf.nn.leaky_relu,
                              name="conv_10")
        x_ = tf.layers.batch_normalization(
            x_,
            name="norm_10")
        