import numpy as np
import pandas as pd

def load_dataframe( features_path, classes_path, classes=["AD", "CN", "EMCI", "LMCI"], balance=False, ceil=3000, seed=0, verbose=False ):
    """
    Arguments:
    - features_path: path to .npz file with extracted features
    - classes_path: path to .csv file with images' name and class, corresponding to the extracted features
    - classes: which classes will be kept
    - balance: whether to balance the dataset or not
    - ceil: maximum number of examples desired in each class (used for balancing)
    - seed: np.random.seed
    - verbose: whether to print dataframe size info
    
    Return:
    - df: dataframe with extracted features
    """
    np.random.seed( seed )

    # LOAD FEATURES

    ## load from .npz
    dataset = np.load( features_path )
    dataset = dataset[ dataset.files[0] ]
    if verbose:
        print( "Number of features: ", dataset.shape[-1] )

    # LOAD CLASSES
    dataset_classes = np.genfromtxt( classes_path, delimiter=',', dtype='str')
    if verbose:
        print( "Original number of examples: ", dataset_classes.shape[0] )

    # CREATE DATAFRAME
    df = pd.DataFrame( dataset )
    df["class"] = dataset_classes[:,1]
    df["img_name"] = dataset_classes[:,0]

    df = df.loc[ df["class"].isin( classes ) ]
    
    if not balance:
        if verbose:
            print( "Final number of examples: ", df.shape[0] )
        return df
    
    # BALANCE DATAFRAME
    index = []
    for c in classes:
        cindex = df.loc[ df["class"] == c ].index.tolist()
        if( len(cindex) > ceil ):
            cindex = np.random.choice( np.array(cindex), ceil, replace=False ).tolist()
        index += cindex

    np.random.shuffle( np.array(index) )
    df = df.loc[ index ]
    
    if verbose:
        print( "Final number of examples: ", df.shape[0] )
    return df