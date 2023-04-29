import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

class ImageDataSet:
    def __init__(self, data_dir):
        train_file_path = os.path.join(data_dir, "Y_train.csv")
        self.labels_train = pd.read_csv(train_file_path, index_col=0).T
        self.data_train = self.load_dataset(os.path.join(data_dir, "X_train"))
        self.data_test = self.load_dataset(os.path.join(data_dir, "X_test"))
    
    def load_dataset(self, dataset_dir):
        # Check if the directory exists
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Directory '{dataset_dir}' does not exist.")

        # Get a list of all the image files in the directory
        image_files = list(Path(dataset_dir).glob("*.png"))

        # Check if there are any images in the directory
        if not image_files:
            raise ValueError(f"No image files found in '{dataset_dir}'.")

        # Note: It's very important to load the images in the correct numerical order!
        dataset_list = [cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE) for image_file in sorted(image_files, key=lambda filename: int(filename.name.rstrip(".png")))]
        return np.stack(dataset_list, axis=0)


