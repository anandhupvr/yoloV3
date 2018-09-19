# import tensorflow as tf
# from matplotlib import pyplot as plt
import numpy as np
# from network.darknet import Arch
import config.parameters as p
# from loss import loss
# import cv2
from utils import datamaker


config = p.getParams()
datamaker.get_data(config) 