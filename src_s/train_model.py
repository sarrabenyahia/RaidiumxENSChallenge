import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from model import build_vgg16_unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

def preprocess_data(data, labels):
    # Resize images to (512, 512)
    data = np.array([cv2.resize(img, (512, 512)) for img in data])

    # Normalize pixel values to [0, 1]
    data = data / 255.0

    # Convert labels to one-hot encoding
    num_classes = 5
    labels = np.eye(num_classes)[labels]

    return data, labels

if __name__ == "__main__":
    # Load the data
    data_dir = Path("./../data/")
    data_train = load_dataset(data_dir / "X_train")
    labels_train = pd.read_csv(data_dir  / "Y_train.csv", index_col=0).T
    
    # Preprocess the data
    data_train, labels_train = preprocess_data(data_train, labels_train)

    # Build the model
    input_shape = (512, 512, 1)
    model = build_vgg16_unet(input_shape)

    # Compile the model
    optimizer = Adam(lr=1e-4)
    loss = categorical_crossentropy
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    epochs = 10
    history = model.fit(data_train, labels_train, epochs=epochs)

    # Save the model
    model.save("my_model.h5")
