from glob import glob
import os
from PIL import Image
import numpy as np
import pydicom

dataset_path = "/home/gama/Documentos/datasets/MRI_AX/"
csv = dataset_path + "MRI_AX_5_14_2022.csv"
# get all files
files = sorted(glob(dataset_path + "ADNI/*/*/*/*/*.dcm"))
print("number of files:", len(files))

print("removing calibration scans...")
files = [k for k in files if "/Calibration_Scan/" not in k]
print("number of files:", len(files))


# sub = [k for k in files if "002_S_0295" in k]

# for subt in sub:
#     print(subt, "\n")


split = files[0].split("/")

old_cfg = split[7] + split[8] + split[9]
old_folder = split[10]

new_files = []
new_files = files

# print("getting only first foler...")
# for file in files:
#     split = file.split("/")

#     cfg = split[7] + split[8] + split[9]
#     folder = split[10]
#     # print("config:", cfg)
#     # print("folder:", folder)
#     if cfg != old_cfg:
#         old_folder = folder
#         old_cfg = cfg

#     if folder != old_folder:
#         # print("folder diff:", folder)
#         continue

#     # print("adding file")
#     new_files.append(file)
#     # input()

print("number of files:", len(new_files))

png_dir = dataset_path + "ADNI_PNG/"
os.makedirs(png_dir, exist_ok=True)

# ADNI_006_S_4150_MR_3_Plane_Localizer__br_raw_20110809103218384_1_S117931_I249410 is multiple frames

for file in new_files:
    print("reading file: ", png_dir + file.split("/")[-1][:-4] + ".png")

    ds = pydicom.dcmread(file)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image.save(png_dir + file.split("/")[-1][:-4] + ".png")
