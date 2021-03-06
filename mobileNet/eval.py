from statistics import mode
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Concatenate
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


def plot_confusion_matrix(y_true, y_pred, classes, save=False, file=None):
    pd.options.display.float_format = "{:,.4f}".format

    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    all = len(y_true)

    acc = [(all + 2 * con_mat[i, i] - np.sum(con_mat[i, :]) - np.sum(con_mat[:, i])) / all for i in range(len(classes))]
    sen = [con_mat[i, i] / np.sum(y_true == i) for i in range(len(classes))]
    esp = [
        (all + con_mat[i, i] - np.sum(con_mat[i, :]) - np.sum(con_mat[:, i])) / np.sum(~(y_true == i))
        for i in range(len(classes))
    ]

    results = pd.DataFrame(
        data=np.array([classes, acc, sen, esp]).T, columns=["class", "Acur??cia", "Sensitividade", "Especificidade"]
    )

    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save:
        plt.savefig(file, bbox_inches="tight", dpi=300, pad_inches=0)
    plt.show()

    results[["Acur??cia", "Sensitividade", "Especificidade"]] = results[["Acur??cia", "Sensitividade", "Especificidade"]].astype(
        float
    )

    print(results)


# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label="%s (AUC:%0.2f)" % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, "b-", label="Random Guessing")
    return roc_auc_score(y_test, y_pred, average=average)


image_path = "/home/gama/Documentos/datasets/MRI_AX/ADNI_PNGv2/"

train_ds, val_ds, test_ds, class_weights = loadDataset(image_path)

model_path = "w_CNN_batch16_low_res_noDataAug"
# model_path = "w_CNN_batch16_low_res_DataAug"
# model_path = "w_CNN_batch4_normal_res_noDataAug"
# model_path = "w_CNN_batch4_normal_res_DataAug"

model = keras.models.load_model(model_path + ".h5")
# evaluate the model
print("\nEvaluate model")
loss, acc = model.evaluate(test_ds, verbose=1)
print("Test Accuracy: %.4f" % acc)
y_pred = model.predict(test_ds)

y_test = np.concatenate([y for x, y in test_ds], axis=0)

plot_confusion_matrix(y_test, np.argmax(y_pred, axis=1), list(class_code.keys()), save=True, file=model_path + "_cm.png")

target = list(class_code.keys())

# set plot figure size
fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))


print("ROC AUC score:", multiclass_roc_auc_score(y_test, y_pred))

c_ax.legend()
c_ax.set_xlabel("False Positive Rate")
c_ax.set_ylabel("True Positive Rate")
plt.savefig(model_path + "roc.png", bbox_inches="tight", dpi=300, pad_inches=0)

plt.show()
