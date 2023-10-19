import numpy as np
import tensorflow as tf


# Generate a sine wave dataset
x = np.linspace(0, 50, 500)
y = np.sin(x)
y = np.reshape(y, (500, 1, 1))
x = np.reshape(x, (500, 1, 1))

# Define an RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(10, activation='relu', input_shape=(1, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x[:-50], y[:-50], epochs=50, validation_data=(x[-50:], y[-50:]))
