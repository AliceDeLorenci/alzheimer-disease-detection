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


def plot_confusion_matrix(y_true, y_pred, classes, save=False, file=None):
    pd.options.display.float_format = "{:,.4f}".format

    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    all = len(y_true)
    
    acc = [(all + 2*con_mat[i,i] - np.sum(con_mat[i,:])-np.sum(con_mat[:,i]))/all for i in range(len(classes))]
    sen = [con_mat[i,i]/np.sum(y_true==i) for i in range(len(classes))]
    esp = [(all + con_mat[i,i] - np.sum(con_mat[i,:]) - np.sum(con_mat[:,i]))/np.sum(~(y_true==i)) for i in range(len(classes))]
    
    results = pd.DataFrame( data = np.array([classes, acc, sen, esp]).T, columns = ["class", "Acurácia" , "Sensitividade", "Especificidade"] )

    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save:
        plt.savefig(file, bbox_inches="tight", dpi=300, pad_inches=0)
    plt.show()

    results[["Acurácia" , "Sensitividade", "Especificidade"]] = results[["Acurácia" , "Sensitividade", "Especificidade"]].astype(float)
    print(results)

