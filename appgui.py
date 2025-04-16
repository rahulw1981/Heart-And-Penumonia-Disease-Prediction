import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from fpdf import FPDF

class DiseaseDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Disease Detection")
        # Increase height to ensure enough space for all widgets
        master.geometry("600x800")
        master.configure(bg="#f0f0f0")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        self.create_widgets()

        # Load the model
        self.model = load_model('best_model.h5')
        self.prediction = None  # to store prediction result
        self.confidence = None  # to store confidence
        self.image_path = None  # selected image path

    def configure_styles(self):
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 12), background='#4CAF50', foreground='white')
        self.style.map('TButton', background=[('active', '#45a049')])
        self.style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Arial', 24, 'bold'), background='#f0f0f0')
        self.style.configure('Result.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        self.style.configure('File.TLabel', font=('Arial', 10), background='#f0f0f0')

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Disease Detection", style='Header.TLabel').pack(pady=20)

        self.image_frame = ttk.Frame(main_frame, width=300, height=300)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(0)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.file_label = ttk.Label(main_frame, text="No file selected", style='File.TLabel')
        self.file_label.pack(pady=5)

        # Container frame for buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.load_button = ttk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        # Place the download button near the other buttons so it's always visible
        self.download_button = ttk.Button(main_frame, text="Download PDF", command=self.download_pdf, state=tk.DISABLED)
        self.download_button.pack(pady=10)

        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack(pady=10)

        self.result_label = ttk.Label(main_frame, text="", style='Result.TLabel')
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.file_label.config(text=f"File: {os.path.basename(file_path)}")
            self.predict_button.config(state=tk.NORMAL)
            self.result_label.config(text="")
            self.download_button.config(state=tk.DISABLED)

    def predict(self):
        self.progress.start()
        self.predict_button.config(state=tk.DISABLED)
        self.master.after(100, self.run_prediction)

    def run_prediction(self):
        img = Image.open(self.image_path).resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        self.prediction = 'Pneumonia' if prediction[0][0] > 0.5 else 'Non Pneumonia'
        self.confidence = float(prediction[0][0] if self.prediction == 'Pneumonia' else 1 - prediction[0][0])

        self.progress.stop()
        self.predict_button.config(state=tk.NORMAL)
        # Enable download button now that we have a result
        self.download_button.config(state=tk.NORMAL)

        result_color = '#FF5252' if self.prediction == 'Pneumonia' else '#4CAF50'
        self.result_label.config(text=f"Prediction: {self.prediction}\nConfidence: {self.confidence:.2f}", foreground=result_color)

    def download_pdf(self):
        if not self.image_path or self.prediction is None:
            messagebox.showerror("Error", "No prediction to generate report for.")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Disease Detection Report", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Prediction: {self.prediction}", ln=True)
        pdf.cell(0, 10, f"Confidence: {self.confidence:.2f}", ln=True)
        pdf.ln(10)
        try:
            pdf.image(self.image_path, x=10, y=pdf.get_y(), w=100)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add image to PDF: {e}")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if save_path:
            try:
                pdf.output(save_path)
                messagebox.showinfo("Success", f"PDF saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save PDF: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    gui = DiseaseDetectionGUI(root)
    root.mainloop()
