import os
import random
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(1)

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224

WEIGHTS_PATH = 'model_weights.h5'

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("Loaded existing model weights.")
else:
    model.save_weights(WEIGHTS_PATH)
    print("Saved initial model weights.")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_array = preprocess_image(file.read())
        prediction = model.predict(img_array)
        class_label = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'
        confidence = float(prediction[0][0])
        return jsonify({'class_label': class_label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
