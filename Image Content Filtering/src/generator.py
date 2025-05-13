import requests
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO

def load_image_from_url(url, label, img_size, num_classes):
    try:
        response = requests.get(url, timeout=10)
        image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img.astype('float32') / 255.0
        return img, tf.one_hot(label, num_classes)
    except:
        return None, None

def data_generator(data_list, batch_size, img_size, num_classes):
    while True:
        batch_images = []
        batch_labels = []
        for url, label in data_list:
            img, lbl = load_image_from_url(url, label, img_size, num_classes)
            if img is not None:
                batch_images.append(img)
                batch_labels.append(lbl)
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_labels)
                batch_images, batch_labels = [], []
