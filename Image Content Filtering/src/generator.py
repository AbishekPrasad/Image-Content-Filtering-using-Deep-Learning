import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

def load_image_from_url(url, label, img_size, num_classes):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(img_size)
        img = np.array(img) / 255.0
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
