import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = 'url_image_classifier.h5' 
IMG_SIZE = (224, 224)
CLASS_NAMES = ['drawing', 'porn', 'sexy', 'neutral', 'hentai']  

model = load_model(MODEL_PATH)

def preprocess_image(image_path, img_size):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image loading failed.")

        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
      
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_image(image_path):
    image = preprocess_image(image_path, IMG_SIZE)
    
    if image is None:
        return "Failed to load or preprocess image."

    prediction = model.predict(image)[0]]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return f"Prediction: {predicted_class} ({confidence * 100:.2f}% confidence)"

if __name__ == '__main__':
    test_image_path = "test_image.jpg"
    result = predict_image(test_image_path)
    print(result)
