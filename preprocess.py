import os
import sys
from glob import glob
import cv2 as cv
from PIL import Image
import numpy as np
import pydicom
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

from skullScrip import skullStripping

import time

dataset_path = "/home/gama/Documentos/datasets/MRI_AX/"
csv_path = dataset_path + "MRI_AX_5_14_2022.csv"

csv_file = pd.read_csv(csv_path)

info = [csv_file["Image Data ID"], csv_file["Group"]]

# get all files
files = sorted(glob(dataset_path + "ADNI/*/*/*/*/*.dcm"))
print("number of files:", len(files))

print("removing calibration scans...")
files = [k for k in files if "/Calibration_Scan/" not in k]
print("number of files:", len(files))

new_files = files

png_dir = dataset_path + "ADNI_PNG/"
os.makedirs(png_dir, exist_ok=True)

id = 0
for file in tqdm(new_files):
    id = file.split("_")[-1][:-4]
    idx = csv_file.index[csv_file["Image Data ID"] == id].to_list()[0]
    filename = png_dir + file.split("/")[-1][:-4] + "_" + csv_file["Group"][idx] + ".png"

    # converts to png
    # start = time.time()
    ds = pydicom.dcmread(file)
    new_image = ds.pixel_array.astype(np.float32)
    scaled_image = np.maximum(new_image, 0) / new_image.max()
    # end = time.time()

    # print("dcm to png:", end - start)
    # start = time.time()
    img = skullStripping(scaled_image)
    # end = time.time()
    # print("skull:", end - start)

    # bias process
    # start = time.time()
    imgSitk = sitk.GetImageFromArray(img.astype(np.float32))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(imgSitk)
    log_bias_field = corrector.GetLogBiasFieldAsImage(imgSitk)
    corrected_image_full_resolution = imgSitk / sitk.Exp(log_bias_field)

    imgSitk = sitk.GetArrayFromImage(corrected_image_full_resolution)
    # end = time.time()
    # print("bias:", end - start)

    # converts to rgb
    # start = time.time()
    rgb_img = np.zeros((3, imgSitk.shape[0], imgSitk.shape[1]))
    rgb_img[::] = imgSitk
    rgb_img = np.transpose(rgb_img, [1, 2, 0])
    # end = time.time()
    # print("2rgb:", end - start)

    # saves image
    cv.imwrite(filename, rgb_img * 255)
