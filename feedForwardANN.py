import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Load and preprocess data
data = pd.read_csv('game_data.csv')  # Replace with your actual data file
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

# Save the model
model.save('game_price_model.h5')
