from discriminator import model as D
from generator import model as G
from discriminator import loss as Dloss
from generator import loss as Gloss
import time
import threading

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.config.list_physical_devices()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
plt.ion()

batch_size = 16
image_dir = "./Changed_3"

g = G(batch_size)
d = D(batch_size)
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = tf.keras.utils.image_dataset_from_directory(image_dir,shuffle=False, labels=None, batch_size=batch_size, image_size=(256, 256)).cache().prefetch(1500)



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  
  os.mkdir("./TrainingData/epoch_{:04d}".format(epoch))
  predictions = (model(test_input, training=False)+1)/2
  print("max: ", np.max(predictions), " min: ", np.min(predictions))
  for i in range(predictions.shape[0]):
      plt.clf()
      plt.imshow(predictions[i])
      plt.axis('off')
      plt.savefig('./TrainingData/epoch_{:04d}/{i}.png'.format(epoch, i=i))
      plt.show(block=False)
      plt.pause(0.02)

  

GenOptimizer = tf.keras.optimizers.Adam(1e-4)
DiscOptimizer = tf.keras.optimizers.Adam(4e-4)

resizeReal = tf.keras.layers.Rescaling(scale=1./255)

@tf.function
def trainBoth(images):
    noise = tf.random.normal([batch_size, 1000])
    gen_loss, disc_loss = 0, 0

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = g(noise, training=True)

      real_output = d(resizeReal(images), training=True)
      fake_output = d(generated_images, training=True)

      gen_loss = Gloss(fake_output)
      disc_loss = Dloss(real_output, fake_output)
      #print("Gen loss: ", gen_loss.numpy(), " Disc loss: ", disc_loss.numpy())
      gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)

      GenOptimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
      DiscOptimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
    return gen_loss, disc_loss

@tf.function
def trainGen(images):
    noise = tf.random.normal([batch_size, 1000])
    gen_loss, disc_loss = 0, 0

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = g(noise, training=True)
      fake_output = d(generated_images, training=True)

      gen_loss = Gloss(fake_output)
      gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)

      GenOptimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    return gen_loss, disc_loss

epochs = 1000
datasetSize = dataset.cardinality().numpy()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=GenOptimizer,
                                 discriminator_optimizer=DiscOptimizer,
                                 generator=g,
                                 discriminator=d
                                 )
noise = tf.random.normal([32, 1000])

epochTimes = []
genLoss = []
discLoss = []


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



for epoch in range(epochs):
  #threading.Thread(target=generate_and_save_images, args=(g, epoch, noise)).start()
  generate_and_save_images(g, epoch, noise)
  if (epoch) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
  stepTimes = np.array([])  
  beginEpoch = time.time()
  for batch_num, image_batch in enumerate(dataset):
    batchBegin = time.time()
    gen_loss, disc_loss = trainBoth(image_batch)
    stepTimes = np.append(stepTimes, time.time()-batchBegin)
    print(f"Epoch: {epoch}/{epochs} Batch: {batch_num}/{datasetSize} Gen Loss: {round(gen_loss.numpy().item(), 4)} Disc Loss: {round(disc_loss.numpy().item(), 4)} Time Elapsed: {round(time.time()-beginEpoch)}s Average Step: {np.average(stepTimes)}s          ", end="\r")
    
  epochTimes.append(time.time()-beginEpoch)
  print("\n")
  genLoss.append(gen_loss.numpy().item())
  discLoss.append(disc_loss.numpy().item())
  #todo: average losses for each epoch not just take the last batch
  plt.clf()
  plt.plot(epochTimes, label="Time")
  plt.xlabel("Epoch")
  plt.ylabel("Time (s)")
  plt.legend()
  plt.savefig("./TrainingData/epochTimes.png")

  plt.clf()
  plt.plot(genLoss, label="Generator Loss")
  plt.plot(discLoss, label="Discriminator Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("./TrainingData/Losses.png")
print(epochTimes)