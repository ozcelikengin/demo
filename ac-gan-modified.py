from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)
num_classes = 10

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()

    cnn.add(Dense(3*3*384, inpit_dim=latent_size, activation='relu'))
    cnn.add(Reshape((3,3,384)))

    #unsample to (7,7,...)
    cnn.add(Conv2DTranspose(192,5, strides=1, padding='valid', activation='relu',kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    #unsample to (14, 14, ...)
    cnn.add(Conv2DTranspose(96,5, strides=2, padding='same', activation='relu',kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    #unsample to (28,28, ...)
    cnn.add(Conv2DTranspose(1,5, strides=2, padding='same', activation='relu',kernel_initializer='glorot_normal'))
    
    #this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size, ))

    #this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Embedding(num_classes, latent_size, embedding_initializer='glorot_normal')(image_class)

    #hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

return Model([latent, image_class], fake_image)

def build_discriminator():
    cnn = Sequential()

    cnn.add(Conv2D(32,3, padding='same', strides=2, input_shape=(28,28,1)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64,3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))
    
    cnn.add(Conv2D(128,3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256,3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28,28,1))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

return Model(image, [fake, aux])

if __name__ == '__main__':
    # batch and latent size take from paper
    epoch = 100
    batch_size = 100
    latent_size = 100

    # Adam parameters
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(optimizer = Adam(learning_rate = adam_lr, beta_1 = adam_beta_1), loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    #
