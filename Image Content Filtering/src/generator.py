import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import cv2
import random

class ImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_data, batch_size, target_size=(128, 128)):
        self.image_data = image_data
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return int(np.floor(len(self.image_data) / self.batch_size))

    def __getitem__(self, index):
        batch_data = self.image_data[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = [], []

        for item in batch_data:
            try:
                response = requests.get(item['url'])
                if response.status_code != 200:
                    continue
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image = np.array(image)

                image = self.preprocess(image)
                images.append(image)
                labels.append(item['label'])
            except Exception:
                continue

        return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=5)

    def preprocess(self, image):
        image = cv2.resize(image, self.target_size)
        if random.random() > 0.5:
            image = np.fliplr(image)
        angle = random.randint(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        return image / 255.0
