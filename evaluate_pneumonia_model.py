import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained model
model = load_model('best_model.h5')

# Image Data Generator for test set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load test data
test_data = test_datagen.flow_from_directory(
    'C:/Users/abhij/Downloads/Heart and Pneumonia/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc}")

# Generate predictions
predictions = model.predict(test_data)
predictions = (predictions > 0.5).astype(int)

# Classification Report
print(classification_report(test_data.classes, predictions, target_names=['Normal', 'Pneumonia']))

# Confusion Matrix
cm = confusion_matrix(test_data.classes, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
