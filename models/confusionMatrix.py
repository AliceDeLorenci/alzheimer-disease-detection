"""
SCC0276 — Machine Learning

Project: Alzheimer disease detection

Authors:
- Alice Valença De Lorenci - 11200289
- Gabriel Soares Gama - 10716511
- Marcos Antonio Victor Arce - 10684621
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_confusion_matrix(y_true, y_pred, classes, save=False, path=None):
    """
    Arguments:
    - y_true: true labels
    - y_pred: predicted labels
    - classes: classes ordered as encoded
    - save: whether to save the confusion matrix
    - path: file where the confusion matrix will be saved if <save> is set to True
    """

    pd.options.display.float_format = "{:,.4f}".format

    # compute confusion matrix and normalize
    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    all = len(y_true)
    
    # accuracy: (TP+TN)/ALL
    acc = [(all + 2*con_mat[i,i] - np.sum(con_mat[i,:])-np.sum(con_mat[:,i]))/all for i in range(len(classes))]
    # sensitivity: TP/(TP+FN)
    sen = [con_mat[i,i]/np.sum(y_true==i) for i in range(len(classes))]
    # specificity: TN/(TN+FP)
    esp = [(all + con_mat[i,i] - np.sum(con_mat[i,:]) - np.sum(con_mat[:,i]))/np.sum(~(y_true==i)) for i in range(len(classes))]
    

    results = pd.DataFrame( data = np.array([classes, acc, sen, esp]).T, columns = ["class", "Acurácia" , "Sensitividade", "Especificidade"] )

    # plot heatmap
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save:
        plt.savefig(path, bbox_inches="tight", dpi=300, pad_inches=0)
    plt.show()

    results[["Acurácia" , "Sensitividade", "Especificidade"]] = results[["Acurácia" , "Sensitividade", "Especificidade"]].astype(float)
    print(results)

