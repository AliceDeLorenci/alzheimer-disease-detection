# needed libraries
from cgi import test
from itertools import count
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model


# tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Concatenate, Conv2D
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from livelossplot import PlotLossesKeras


# sklearn
import sklearn
import sklearn.model_selection
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

# custom libs
import sys


# plot
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import random
from glob import glob

from dataloader import *


np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


# load full model
cnn_encoder = MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling="max")
x = Dense(4, activation="Softmax")(cnn_encoder.output)
cnn_model = Model(inputs=cnn_encoder.inputs, outputs=x)
# confirms that the parameters are trainable
for layer in cnn_model.layers:
    layer.trainable = True

image_path = "/home/gama/Documentos/datasets/MRI_AX/ADNI_PNGv2/"
train_ds, val_ds, test_ds, class_weights = loadDataset(image_path, aug=False, batch_size=16)

# for img, label in train_ds:
#     print(img.shape)
#     print(label)
#     print(img.cpu().numpy().shape)
#     # cv.imshow("img", img.cpu().numpy()[0])
#     # cv.waitKey(0)
#     break

print(cnn_model.summary())
# compile the model
initial_learning_rate = 0.0001
opt = keras.optimizers.Adam(learning_rate=0.0001)

cnn_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

tb_callback = tf.keras.callbacks.TensorBoard("./logs/w_batch4_normal_res_noDataAug", update_freq=1)

monitor_val_acc = EarlyStopping(monitor="val_loss", patience=3)
history = cnn_model.fit(
    train_ds, validation_data=val_ds, epochs=15, verbose=1, callbacks=[tb_callback, monitor_val_acc], class_weight=class_weights
)

cnn_model.save("w_CNN_batch4_normal_res_noDataAug.h5")
