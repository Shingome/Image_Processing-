import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import shutil


dataset_arr: np.ndarray = np.load("../dataset.npy", allow_pickle=True)
dataset_path = "../dataset/"

if os.path.exists(os.path.dirname(dataset_path)):
    shutil.rmtree(dataset_path)

os.makedirs(dataset_path)

iteration = 0
for y, x in tqdm(dataset_arr):
    x = np.reshape(x[0] * 255, (8, 8))
    img = Image.fromarray(x).convert("1")
    save_path = dataset_path + str(y)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = save_path + "/{0}.jpg".format(f"{iteration}".zfill(9))
    img.save(filename)
    iteration += 1
