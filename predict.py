import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model = load_model('best_model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Rescale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make prediction
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = 'Pneumonia' if prediction[0] > 0.5 else 'Non Pneumonia'
    return predicted_class


image_path = 'C:/Users/abhij/Downloads/Heart and Pneumonia/41479_2016_5010038_Fig1.jpg'  # Replace with the path to your image
result = predict_image(image_path)
print(f'Predicted Class: {result}')
