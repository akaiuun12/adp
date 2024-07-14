# %%
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the autoencoder model
def build_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)
    
    # Encoder
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(decoded)
    
    # Autoencoder model
    autoencoder = models.Model(input_img, decoded)
    
    return autoencoder

# Example usage
input_shape = (784,)  # For MNIST data (flattened 28x28 images)
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
# (Assuming X_train contains your training data)
# autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
