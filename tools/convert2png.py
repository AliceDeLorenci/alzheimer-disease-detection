from glob import glob
import os
from PIL import Image
import numpy as np
import pydicom
import pandas as pd
from tqdm import tqdm

dataset_path = "/home/gama/Documentos/datasets/MRI_AX/"
csv_path = dataset_path + "MRI_AX_5_14_2022.csv"

csv_file = pd.read_csv(csv_path)
# print(csv_file.head())

info = [csv_file["Image Data ID"], csv_file["Group"]]

# get all files
files = sorted(glob(dataset_path + "ADNI/*/*/*/*/*.dcm"))
print("number of files:", len(files))

print("removing calibration scans...")
files = [k for k in files if "/Calibration_Scan/" not in k]
print("number of files:", len(files))

new_files = []
new_files = files

print("number of files:", len(new_files))

png_dir = dataset_path + "ADNI_PNG/"
os.makedirs(png_dir, exist_ok=True)

id = 0
for file in tqdm(new_files):
    # print("file split: ", file.split("/")[-1][:-4])
    # print("id split: ", file.split("_")[-1][:-4])
    id = file.split("_")[-1][:-4]
    idx = csv_file.index[csv_file["Image Data ID"] == id].to_list()[0]
    filename = (
        png_dir + file.split("/")[-1][:-4] + "_" + csv_file["Group"][idx] + ".png"
    )
    # print("filename: ", str(filename))

    ds = pydicom.dcmread(file)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image.save(png_dir + file.split("/")[-1][:-4] + ".png")
