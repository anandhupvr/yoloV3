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

    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    def network(self):
            # Layer 1
        conv1 = tf.layers.conv2d(
            self.x,
            filters=32,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_1")
        conv1 = tf.layers.batch_normalization(
            conv1,
            name="norm_1")
        maxpool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")
        # Layer 2

        conv2 = tf.layers.conv2d(
            maxpool1,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_2")
        conv2 = tf.layers.batch_normalization(
            conv2,
            name="norm_2")
        maxpool2 = tf.nn.max_pool(
            conv2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")
        # Layer 3

        conv3 = tf.layers.conv2d(
            maxpool2, filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_3")
        conv3 = tf.layers.batch_normalization(
            conv3,
            name="norm_3")

        # Layer 4
        conv4 = tf.layers.conv2d(
            conv3, filters=64,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_4")
        conv4 = tf.layers.batch_normalization(
            conv4,
            name="norm_4")

        # Layer 5
        conv5 = tf.layers.conv2d(
            conv4,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_5")
        conv5 = tf.layers.batch_normalization(
            conv5,
            name="norm_5")
        max_pool3 = tf.nn.max_pool(
            conv5,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

        # Layer 6
        conv6 = tf.layers.conv2d(
            max_pool3,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_6")
        conv6 = tf.layers.batch_normalization(
            conv6,
            name="norm_6")

        # Layer 7
        conv7 = tf.layers.conv2d(
            conv6,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_7")
        conv7 = tf.layers.batch_normalization(
            conv7,
            name="norm_7")

        # Layer 8
        conv8 = tf.layers.conv2d(
            conv7,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_8")
        conv8 = tf.layers.batch_normalization(
            conv8,
            name="norm_8")
        max_pool4 = tf.nn.max_pool(
            conv8,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

        # Layer 9
        conv9 = tf.layers.conv2d(
            max_pool4,
            filters=512,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_9")
        conv9 = tf.layers.batch_normalization(
            conv9,
            name="norm_9")

        # Layer 10
        conv10 = tf.layers.conv2d(
            conv9,
            filters=256,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_10")
        conv10 = tf.layers.batch_normalization(
            conv10,
            name="norm_10")

        # Layer 11
        conv11 = tf.layers.conv2d(
            conv10,
            filters=512,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_11")
        conv11 = tf.layers.batch_normalization(
            conv11,
            name="norm_11")

        # Layer 12
        conv12 = tf.layers.conv2d(
            conv11,
            filters=256,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_12")
        conv12 = tf.layers.batch_normalization(
            conv12,
            name="norm_12")

        # Layer 13
        conv13 = tf.layers.conv2d(
            conv12,
            filters=512,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_13")
        conv13 = tf.layers.batch_normalization(
            conv12,
            name="norm_13")

        skip_connection = conv13

        max_pool5 = tf.nn.max_pool(
            conv13,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

        # Layer 14
        conv14 = tf.layers.conv2d(
            max_pool5,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_14")
        conv14 = tf.layers.batch_normalization(
            conv14,
            name="norm_14")

        # Layer 15
        conv15 = tf.layers.conv2d(
            conv14,
            filters=512,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_15")

        conv15 = tf.layers.batch_normalization(
            conv15,
            name="norm_15")

        # Layer 16
        conv16 = tf.layers.conv2d(
            conv15,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_16")
        conv16 = tf.layers.batch_normalization(
            conv16,
            name="norm_16")

        # Layer 17
        conv17 = tf.layers.conv2d(
            conv16,
            filters=512,
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_17")
        conv17 = tf.layers.batch_normalization(
            conv16,
            name="norm_17")

        # Layer 18
        conv18 = tf.layers.conv2d(
            conv17,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_18")
        conv18 = tf.layers.batch_normalization(
            conv18,
            name="norm_18")

        # Layer 19
        conv19 = tf.layers.conv2d(
            conv18,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_19")
        conv19 = tf.layers.batch_normalization(
            conv19,
            name="norm_19")

        # Layer 20
        conv20 = tf.layers.conv2d(
            conv19,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_20")
        conv20 = tf.layers.batch_normalization(
            conv20,
            name="norm_20")

        # Layer 21

        conv21 = tf.layers.conv2d(
            skip_connection,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_21")
        skip_connection = tf.layers.batch_normalization(
            conv21,
            name="norm_21")
        # skip_connection = space_to_depth_x2(skip_connection)
        skip_connection = tf.space_to_depth(skip_connection, block_size=2)

        route = tf.concat(
            [skip_connection,
             conv20],
             3)

        # Layer 22

        conv22 = tf.layers.conv2d(
            route,
            filters=1024,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_22")
        conv22 = tf.layers.batch_normalization(
            conv22,
            name="norm_22")

        # Layer 23
        conv23 = tf.layers.conv2d(
            conv22,
            filters=self.config["BOX"] * (4 + 1 + self.config["CLASS"]),
            kernel_size=[1, 1],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_23")
        output = tf.reshape(
            conv23,
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
