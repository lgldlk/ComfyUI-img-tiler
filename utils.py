
import os




def get_all_directories(path):
    directories = []
    # os.walk 会遍历指定路径下的所有文件和子文件夹
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            directories.append(os.path.join(root, dir))
    return directories
  
  



import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

def torch_tensor_to_cv2_image(tensor_image):
    tensor_image=tensor_image.numpy()
    modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))
    modified_image = np.array(modified_image).astype(np.float32)
    modified_image = np.clip(modified_image, 0, 255) / 255
    hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
    return hls_img


def cv2_image_to_torch_tensor(cv2_image):
    # Convert the cv2 image to a numpy array
    numpy_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    numpy_image = np.transpose(numpy_image, (2, 0, 1))

    # Convert the numpy array to a tensor
    tensor = torch.from_numpy(numpy_image)
    return tensor