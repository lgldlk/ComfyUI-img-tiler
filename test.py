import os
import re


def get_all_directories(path):
    directories = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            directories.append(os.path.normpath(os.path.join(root, dir)))
    return directories


def find_files_with_extensions(directory, extensions, exclude_strs=None):
    result = []
    for root, dirs, files in os.walk(directory):
        if exclude_strs and any(exclude_str in root for exclude_str in exclude_strs):
            continue
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                result.append(os.path.join(root, file))
    return result


now_folder_path = os.path.dirname(__file__)

tilesList = []
for dir in get_all_directories(os.path.join(now_folder_path, "tiles")):
    dir_res = re.search(r"[\\|\/]gen_(.*)", dir)
    if dir_res:
        tilesList.append({"name": dir_res.group(1), "path": dir, "type": "tile"})

for dir in find_files_with_extensions(
    os.path.join(now_folder_path, "tiles"), "png", ["gen_"]
):
    dir_res = re.search(r"[\\|\/]([^\.\/\\]*)\.png", dir)
    if dir_res:
        tilesList.append({"name": dir_res.group(1), "path": dir, "type": "image"})

print(tilesList)
