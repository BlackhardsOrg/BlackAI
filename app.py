import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('game_data.csv')  # Replace with your actual data file

# Print the column names to verify
print("Columns in the dataset:", data.columns)

# Check if 'price' column exists
if 'price' not in data.columns:
    raise KeyError("The 'price' column does not exist in the dataset. Available columns: " + str(data.columns))

X = data[['previous_price', 'competitor_price', 'sales', 'seasonal_demand']].values
y = data['price'].values

# Define the model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Output layer for price
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Ensure the directory exists or create it
save_dir = './'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'game_price_model.h5')

# Save the model
model.save(save_path)

# Verify the model file is saved
if os.path.exists(save_path):
    print(f"Model saved successfully at {save_path}")
else:
    print("Failed to save the model.")

# Path to the model
model_path = './game_price_model.h5'

# Load the trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
else:
    print(f"Model file not found at {model_path}")

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    print(data, "DATA")
    features = np.array(data['features']).reshape(1, -1)
    predicted_price = model.predict(features)
    return jsonify({'predicted_price': float(predicted_price[0][0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
