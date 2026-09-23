import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

raw_dir = "data/raw/images"
processed_dir = "data/processed/images"

os.makedirs(processed_dir, exist_ok=True)

x_list = []
y_list = []

labels_map = {"ham": 0, "spam": 1}

for folder_name, label in labels_map.items():
    folder_path = os.path.join(raw_dir, folder_name)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        image = cv2.imread(file_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))

        x_list.append(image)
        y_list.append(label)




