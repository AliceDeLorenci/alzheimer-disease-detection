import numpy as np
import pandas as pd

def load_dataframe( features_path, classes_path ):

    # LOAD FEATURES

    ## load from .csv
    # dataset = np.loadtxt( data_path + 'features.csv', delimiter=',' )

    ## load from .npz
    dataset = np.load( features_path )
    dataset = dataset[ dataset.files[0] ]

    # LOAD CLASSES
    classes = np.genfromtxt( classes_path, delimiter=',', dtype='str')
    classes.shape

    # CREATE DATAFRAME
    df = pd.DataFrame( dataset )
    df["class"] = classes[:,1]

    return df