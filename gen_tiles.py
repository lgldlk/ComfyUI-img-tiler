import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import math
import conf
from torchvision import transforms

# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
DEPTH = conf.DEPTH
# list of rotations, in degrees, to apply over the original image
ROTATIONS = conf.ROTATIONS

img_path = sys.argv[1]
img_dir = os.path.dirname(img_path)
img_name, ext = os.path.basename(img_path).rsplit(".", 1)
out_folder = img_dir + "/gen_" + img_name

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = img.astype("float")

height, width, channels = img.shape
center = (width / 2, height / 2)

for b in tqdm(np.arange(0, 1.01, 1 / DEPTH)):
    for g in np.arange(0, 1.01, 1 / DEPTH):
        for r in np.arange(0, 1.01, 1 / DEPTH):
            mult_vector = [b, g, r]
            if channels == 4:
                mult_vector.append(1)
            new_img = img * mult_vector
            new_img = new_img.astype("uint8")
            for rotation in ROTATIONS:
                if isinstance(new_img, np.ndarray):
                    new_img = transforms.ToPILImage()(new_img)
                rotated_img = transforms.functional.rotate(
                    new_img, angle=rotation, expand=True
                )
                rotated_img = np.array(rotated_img)
                new_h, new_w = rotated_img.shape[:2]
                cv2.imwrite(
                    f"{out_folder}/{img_name}_{round(r,1)}_{round(g,1)}_{round(b,1)}_r{rotation}.{ext}",
                    rotated_img,
                    # compress image
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
