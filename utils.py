
import os




def get_all_directories(path):
    directories = []
    # os.walk 会遍历指定路径下的所有文件和子文件夹
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            directories.append(os.path.normpath(os.path.join(root, dir)))
    return directories
  
  




import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255
  
def torch_tensor_to_cv2_image(tensor_image):
    # 1. 去掉 batch 维度 (假设输入形状为 [batch, height, width, channels])
    tensor_image = tensor_image.squeeze(0)

    # 2. 将归一化的 [0, 1] 数据转换为 [0, 255]
    tensor_image = (tensor_image * 255).clamp(0, 255).byte()

    # 3. 将 PyTorch tensor 转换为 numpy 数组
    numpy_image = tensor_image.numpy()

    # 4. 确保通道顺序为 (height, width, channels)，如果输入格式是 (channels, height, width)，则调整维度
    if numpy_image.shape[0] in [1, 3]:  # 通道在第一维
        numpy_image = np.transpose(numpy_image, (1, 2, 0))  # 转换为 (height, width, channels)

    # 5. 如果是 RGB 图像，OpenCV 需要 BGR 格式
    if numpy_image.shape[2] == 3:  # 检查是否为 RGB 图像
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return numpy_image

def cv2_image_to_torch_tensor(cv2_image):
    modified_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

    modified_image = modified_image.astype(np.uint8)
    modified_image = modified_image / 255
    modified_image = torch.from_numpy(modified_image).unsqueeze(0)

    return modified_image