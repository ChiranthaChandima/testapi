from flask import Flask, request, jsonify
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL = tf.keras.models.load_model('saved_models/1')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.route("/ping", methods=["GET"])
def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    
    if file is None:
        return jsonify({"error": "File not provided"}), 400

    image = read_file_as_image(file.read())

    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return jsonify({
        'class': predicted_class,
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(debug=True)

