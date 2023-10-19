from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.models import Sequential
import numpy as np

# Load and preprocess data
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train - 127.5) / 127.5  # Normalize to [-1, 1]

# Generator
generator = Sequential([
    Dense(128, activation='relu', input_dim=100),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Simplified GAN training loop
epochs = 10000
batch_size = 32

for epoch in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise)
    real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
    
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    
    noise = np.random.normal(0, 1, (batch_size, 100))
    labels_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, labels_gan)
