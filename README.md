This project was developed for the Machine Learning (SSC0276) course at the University of São Paulo (USP) by:
- Alice Valença De Lorenci - 11200289
- Gabriel Soares Gama - 10716511
- Marcos Antonio Victor Arce - 10684621

The MRI dataset used was kindly provided by ADNI and anyone wishing to reproduce our results may apply for [access to the data](https://adni.loni.usc.edu/data-samples/access-data/).

The code is organized in the folders ```data```, ```models``` and ```preprocessing```, and the relevant files are:
- ```preprocessing```:
  - ```preprocessing.ipynb```: preprocessing flow applied to the data
  - ```feature_extraction.ipynb```: feature extraction using a CNN pre-trained on the ImageNet database
  - ```load_dataframe.ipynb```: instructions on how to load a dataframe with the extracted features
- ```data```:
  - ```features.npz```: features extracted from the MRI images
  - ```name_class.csv```: classes of the MRI images
- ```models```:
  - ```KNN.ipynb```: KNN model
  - ```MultilayerPerceptron.ipynb```: multilayer perceptron model
  - ```SVM.ipynb```: SVM model

  
