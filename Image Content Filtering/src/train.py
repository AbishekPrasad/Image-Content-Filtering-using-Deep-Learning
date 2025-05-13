from data_loader import load_data_from_files
from generator import data_generator
from model import build_custom_cnn
import math

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

class_files = {
    'drawing': '../data/urls_drawings.txt',
    'porn': '../data/urls_porn.txt',
    'sexy': '../data/urls_sexy.txt',
    'neutral': '../data/urls_neutral.txt',
    'hentai': '../data/urls_hentai.txt'
}


data, class_to_label = load_data_from_files(class_files)
num_classes = len(class_to_label)

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

train_gen = data_generator(train_data, BATCH_SIZE, IMG_SIZE, num_classes)
val_gen = data_generator(val_data, BATCH_SIZE, IMG_SIZE, num_classes)

model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=num_classes)
model.summary()

model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS
)

model.save('url_image_classifier.h5')

steps_per_epoch = math.floor(len(train_data) / BATCH_SIZE)
validation_steps = math.floor(len(val_data) / BATCH_SIZE)
