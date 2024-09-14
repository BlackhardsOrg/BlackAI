from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('game_price_model.h5')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    predicted_price = model.predict(features)
    return jsonify({'predicted_price': float(predicted_price[0][0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
