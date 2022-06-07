"""
SCC0276 — Machine Learning

Project: Alzheimer disease detection

Authors:
- Alice Valença De Lorenci - 11200289
- Gabriel Soares Gama - 10716511
- Marcos Antonio Victor Arce - 10684621

The following code was reproduced from:
    https://github.com/KaueTND/Brain_Extraction_T1wMRI/blob/main/Workflow_jupyter.ipynb
"""

# libraries
from PIL import Image
from scipy import signal as sci
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
import matplotlib.image as mpimg
from skimage import morphology as morph
from scipy import ndimage as nd
import sys, os
from skimage.filters import (
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_local,
    threshold_minimum,
)
from skimage import feature
from skimage.filters import threshold_local as threshold_adaptative
import heapq
import matplotlib.patches as mpatches
from skimage.morphology import convex_hull_image
import pandas as pd
import seaborn as sns
from scipy.signal import convolve2d as sc_conv
from matplotlib.colors import colorConverter
import matplotlib as mpl
import time
import cv2 as cv


def save_without_spacing(nome, fig):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(nome, bbox_inches="tight", pad_inches=0, dpi=fig.dpi)

# remove background noise
def remove_ruido_fundo(imagem, Ti=0):
    if Ti == 0:
        Ti = np.mean(imagem)
    imagem_filtrada = sci.medfilt(imagem, [3, 3])
    ti = imagem_filtrada > Ti
    mascara = binary_fill_holes(ti)

    return [Ti, imagem * mascara]

# adjust contrast
def ajuste_contraste(img):
    A = img
    L = np.max(A.ravel())
    B = (L - 1) - A
    SE1 = morph.ball(3)[3]
    C = morph.closing(B, SE1)
    L = np.max(C.ravel())
    D = (L - 1) - C
    Adiff = A - D
    return np.clip(A + Adiff, 0, 255)

# image binarization, used to find conex components
def binariza(imagem, Cerebro=True):
    if Cerebro:
        t = threshold_otsu(imagem)
    else:  # Crânio
        t = 0
    imagem_binaria = imagem.copy()
    imagem_binaria[imagem_binaria <= t] = 0
    imagem_binaria[imagem_binaria > t] = 1
    return imagem_binaria

# detect connected component border
def detecta_borda_conexa(imagem_binaria):
    cranio_convexo = convex_hull_image(binary_fill_holes(imagem_binaria))  # preenche a imagem
    borda_cranio = feature.canny(cranio_convexo)  # extrai a borda
    return borda_cranio

# detect connected component
def detecta_componente_conexa(imagem_binaria, mostrar_Passos=False):
    label_aux = morph.label(imagem_binaria)
    label_ret = morph.label(imagem_binaria)
    if mostrar_Passos == True:
        plt.figure()
        plt.subplot(121)
        plt.imshow(label_ret, "flag")
        # plt.title('Componentes Conexas(Labels)')
        # save_without_spacing("TrabalhoMeninge_7_"+str(0)+"comp.png")
    valor_disco = 0

    # Passo 1 - Detecção da Borda do Crânio via Canny
    contorno_cranio_binario = detecta_borda_conexa(imagem_binaria)
    if mostrar_Passos == True:
        plt.subplot(122)
        plt.imshow(contorno_cranio_binario, "gray")
        plt.title("Detecção Borda Canny")
        plt.figure()

    # Passo 2 - Detecção da Tonalidade do Fundo
    tonalidade_fundo = np.argmax(np.bincount(label_ret[0:20, 0:20].ravel()))
    for ab in range(10):
        # print(ab)
        contorno_cranio = label_ret[contorno_cranio_binario == True]
        [IMX, IMY] = imagem_binaria.shape

        # Passo 3 - Detecção da Parte Central do Cérebro (sem fundo)
        array_centro = label_ret[IMX // 2 - 25 : IMX // 2 + 25, IMY // 2 - 25 : IMY // 2 + 25]
        array_centro = array_centro[array_centro != tonalidade_fundo]
        max_centro = np.argmax(
            np.bincount(array_centro.ravel())
        )  # retorna o pixel mais frequente na imagem array_centro, correspondendo ao valor do cérebro

        # Passo 4 - Verificação da Presença da Intensidade da Borda no Centro
        if max_centro in contorno_cranio:
            valor_disco = valor_disco + 1
            label_ret = morph.opening(label_aux, morph.disk(valor_disco))
            label_ret = morph.label(label_ret)
            # plt.imshow(label_ret)
            # save_without_spacing("TrabalhoMeninge_7_"+str(valor_disco)+"comp.png")
        else:
            break

    # print(valor_disco)
    if valor_disco == 0:
        label_ret = label_aux
    return [label_ret, tonalidade_fundo, array_centro]

# extract connected component
def extrai_maior_componente(LC, tonalidade_fundo, array_centro=0, mostrar_Passos=False):
    # print(type(array_centro))
    c1 = LC.copy()

    if mostrar_Passos == True:
        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.imshow(c1, "gray")
        plt.axis("off")
        plt.title("Componentes Conexas")
    hist = np.bincount(LC.ravel())
    index = heapq.nlargest(1, range(len(hist)), key=hist.__getitem__)[-1]

    # Se a maior componente conexa for o fundo, pegue a segunda maior componente
    if index == tonalidade_fundo:
        index = heapq.nlargest(2, range(len(hist)), key=hist.__getitem__)[-1]

    # Rerotulação das componentes conexas baseado na ocorrencia de pixels no centro
    if type(array_centro) == np.ndarray:
        matriz_substituicao = np.isin(c1, np.unique(array_centro[array_centro != tonalidade_fundo]))
        c1_k = convex_hull_image(matriz_substituicao)
        if mostrar_Passos == True:
            plt.figure()
            plt.imshow(matriz_substituicao, "gray")
            plt.axis("off")
            # plt.title('Agrupamento Conexo')
            # save_without_spacing("TrabalhoMeninge_8_Agrupamento.png")
        return [matriz_substituicao, c1_k]
        # c1[matriz_substituicao] = index

    c1[c1 == index] = 1000
    c1[c1 < 1000] = 0
    c1[c1 == 1000] = 1
    c1_k = convex_hull_image(c1)
    return [c1, c1_k]


def skullStripping(im1):
    # im1 = Image.open(fileName)
    img = np.array(im1)
    ruido = three_segments(220, 240, 50, 210)
    ruido = ruido[::-1]
    [media, img] = remove_ruido_fundo(img)
    Aenhanced = ajuste_contraste(img)
    Gbin = binariza(Aenhanced)
    [LC, tonalidade_fundo, array_centro] = detecta_componente_conexa(Gbin, False)
    [c1, c1_k] = extrai_maior_componente(LC, tonalidade_fundo, array_centro, False)

    img = img * c1_k

    return img
