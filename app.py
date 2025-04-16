import os
from io import BytesIO
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from fpdf import FPDF

app = Flask(__name__)

# Load the models
pneumonia_model = load_model('best_model.h5')
heart_model = load_model('heart_mri_model.h5')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Pneumonia prediction utilities
def preprocess_pneumonia_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Rescale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_pneumonia(image_path):
    img_array = preprocess_pneumonia_image(image_path)
    prediction = pneumonia_model.predict(img_array)
    predicted_class = 'Pneumonia' if prediction[0] > 0.5 else 'Non Pneumonia'
    confidence = float(prediction[0][0] if predicted_class == 'Pneumonia' else 1 - prediction[0][0])
    return predicted_class, confidence


# Heart disease prediction utilities
def preprocess_heart_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, img_size)  # Resize to match model input
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, img_size[0], img_size[1], 1)  # Add batch dimension
    return img


def predict_heart_disease(image_path):
    processed_image = preprocess_heart_image(image_path)
    prediction = heart_model.predict(processed_image)
    predicted_class = 'Sick' if prediction[0][0] > 0.5 else 'Normal'
    confidence = float(prediction[0][0] if predicted_class == 'Sick' else 1 - prediction[0][0])
    return predicted_class, confidence


# Updated PDF generation function
def generate_pdf_report(image_path, predicted_class, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Disease Detection Report", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Prediction: {predicted_class}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}", ln=True)
    pdf.ln(10)
    # Add the image to the PDF; adjust x, y, w as needed
    pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
    # Generate the PDF as a string, encode it, and wrap in BytesIO
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_output)
    pdf_buffer.seek(0)
    return pdf_buffer


@app.route('/')
def index():
    return render_template('index3.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check prediction type from the request
        prediction_type = request.form.get('type', 'pneumonia')  # Default to pneumonia
        try:
            if prediction_type == 'pneumonia':
                predicted_class, confidence = predict_pneumonia(filepath)
            elif prediction_type == 'heart':
                predicted_class, confidence = predict_heart_disease(filepath)
            else:
                return jsonify({'error': 'Invalid prediction type'}), 400

            # Check if PDF download is requested
            download_pdf = request.form.get('download_pdf', '').lower() == 'true'
            if download_pdf:
                pdf_buffer = generate_pdf_report(filepath, predicted_class, confidence)
                return send_file(pdf_buffer,
                                 as_attachment=True,
                                 download_name="result.pdf",
                                 mimetype="application/pdf")

            # Otherwise return JSON response with prediction info
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'image_url': f'/static/uploads/{filename}'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
