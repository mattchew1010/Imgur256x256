import tensorflow as tf
from keras import layers

def model(batch_size=16):
    return tf.keras.Sequential([
      #input 1000 numbers of noise
      layers.Dense(2048, input_shape=(1000,)),
      layers.BatchNormalization(),
      layers.LeakyReLU(),

      layers.Dense(2048),
      layers.BatchNormalization(),
      layers.LeakyReLU(),

      layers.Dense(256* 16 * 16),
      layers.BatchNormalization(),
      layers.LeakyReLU(),
      layers.Reshape((16, 16, 256)),

      layers.Conv2DTranspose(256, 3, strides=2, data_format='channels_last', padding='same'),
      layers.BatchNormalization(),
      layers.LeakyReLU(),
      
      layers.Conv2DTranspose(128, 3, strides=2, data_format='channels_last', padding='same'),
      layers.BatchNormalization(),
      layers.LeakyReLU(),

      layers.Conv2DTranspose(64, 3, strides=2, data_format='channels_last', padding='same'),
      layers.BatchNormalization(),
      layers.LeakyReLU(),

      layers.Conv2DTranspose(32, 3, strides=2, data_format='channels_last', padding='same'),
      layers.BatchNormalization(),
      layers.LeakyReLU(),

      #output with 3 filters for R, G, B
      layers.Conv2DTranspose(3, 3, strides=1, padding='same', use_bias=False, activation='tanh'),
      #layers.Rescaling(scale=1./255),
    ])

def loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)