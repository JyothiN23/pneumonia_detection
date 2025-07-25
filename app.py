from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("pneumonia_cnn_model.h5",compile=False)

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    prediction = model.predict(img_tensor)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    filepath = os.path.join('static', file.filename)
    file.save(filepath)
    result = predict_pneumonia(filepath)
    return render_template('result.html', result=result, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
