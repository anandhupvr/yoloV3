import tensorflow as tf
import numpy as np


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
        # Layer 1
        x = tf.layers.conv2d(
            self.x,
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            name="conv_1")
        x = tf.layers.batch_normalization(
        	x,
        	name="norm_1")
        x = tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

       # Layer 2 - 5
        for i in range(0, 4):
            x = tf.layers.conv2d(
                x,
                filters=32*(2**i),
                kernel_size=[3, 3],
                strides=(1, 1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_"+str(i+2))
            x = tf.layers.batch_normalization(
                x,
                name="norm_"+str(i+2))
            x = tf.nn.max_pool(
                x,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME")
        # Layer 6
        x = tf.layers.conv2d(
            x,
            filters=512,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_6")
        x = tf.layers.batch_normalization(
            x,
            name="norm_6")
        x = tf.nn.max_pool(x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

        # Layer 7 - 8

        for i in range(0, 2):
            x = tf.layers.conv2d(
                x,
                filters=1024,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_"+str(i+7))
            x = tf.layers.batch_normalization(
                x,
                name="norm_"+str(i+7))
        output = tf.reshape(
            x,
            [
                self.config["BATCH_SIZE"],
                self.config["GRID_H"],
                self.config["GRID_W"],
                self.config["BOX"],
                4 + 1 + self.config["CLASS"]
            ])
        return output
    def getX(self):
        return self.x