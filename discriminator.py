import tensorflow as tf
from keras import layers

def model(batch_size=16):
   return tf.keras.Sequential([
      #input 256x256x3 image
      #layers.Rescaling(scale=1./255),
      # layers.Flatten(),
      # layers.Dense(1024, activation='sigmoid'),
      # layers.Dense(256, activation='sigmoid'),
      layers.Conv2D(32, 3, strides=(2, 2), padding='same',input_shape=[256, 256, 3]),
      layers.LeakyReLU(),
      layers.Dropout(0.3),

      layers.Conv2D(64, 3, strides=(2, 2), padding='same'),
      layers.LeakyReLU(),
      layers.Dropout(0.3),

      layers.Conv2D(128, 3, strides=(2, 2), padding='same'),
      layers.LeakyReLU(),
      layers.Dropout(0.3),
      
      layers.Conv2D(256, 3, strides=(2, 2), padding='same'),
      layers.LeakyReLU(),
      layers.Dropout(0.3),

      layers.Flatten(),
      layers.Dense(1),
   ])
def loss(real_output, fake_output):
   real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
   fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
   total_loss = real_loss + fake_loss
   return total_loss