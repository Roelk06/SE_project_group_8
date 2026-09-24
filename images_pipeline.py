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

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        x_list.append(image)
        y_list.append(label)

x = np.array(x_list, dtype=np.float32)
y = np.array(y_list)

x = x / 255.0      
x = np.transpose(x, (0, 3, 1, 2))   

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

np.save(os.path.join(processed_dir, "x_train.npy"), x_train)
np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
np.save(os.path.join(processed_dir, "x_val.npy"), x_val)
np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
np.save(os.path.join(processed_dir, "x_test.npy"), x_test)
np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

