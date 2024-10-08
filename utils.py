
import os


def get_all_directories(path):
    directories = []
    # os.walk 会遍历指定路径下的所有文件和子文件夹
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            directories.append(os.path.join(root, dir))
    return directories
  
  
print(get_all_directories('./tiles'))