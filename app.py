from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import io

app = Flask(__name__)

# CIFAR-10 sınıfları
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Taşıt sınıfları
TRANSPORT_CLASSES = ['airplane', 'automobile', 'ship', 'truck']

# Model yükleme
def load_model():
    if not os.path.exists('results/cifar10_model.h5'):
        return None
    return tf.keras.models.load_model('results/cifar10_model.h5')

# Model yükle
model = load_model()

@app.route('/')
def home():
    return render_template('index.html', classes=CIFAR10_CLASSES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
    
    try:
        # Görüntüyü oku ve işle
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Tahmin yap
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        predicted_label = CIFAR10_CLASSES[predicted_class]
        confidence = float(predictions[0][predicted_class])
        
        if predicted_label not in TRANSPORT_CLASSES:
            return jsonify({
                'class': None,
                'confidence': confidence,
                'all_predictions': predictions[0].tolist(),
                'message': 'Bu bir taşıt değildir.'
            })
        else:
            return jsonify({
                'class': predicted_label,
                'confidence': confidence,
                'all_predictions': predictions[0].tolist(),
                'message': None
            })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if model is None:
        print("Hata: Model dosyası bulunamadı! Lütfen önce modeli eğitin.")
    else:
        app.run(debug=True) 