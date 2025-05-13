from data_loader import load_image_data_from_multiple_files
from generator import ImageDataGenerator
from model import create_model

file_paths = [
    '../data/urls_drawings.txt',
    '../data/urls_hentai.txt',
    '../data/urls_neutral.txt',
    '../data/urls_porn.txt',
    '../data/urls_sexy.txt'
]

# Load image data
image_data = load_image_data_from_multiple_files(file_paths)

# Create data generator
batch_size = 32
train_generator = ImageDataGenerator(image_data, batch_size=batch_size)

# Create and train the model
model = create_model()
model.fit(train_generator, epochs=10)
