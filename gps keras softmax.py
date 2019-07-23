import pandas as pd
import numpy as np
import os
import random
import keras
import os
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Input, Dropout, Flatten, BatchNormalization, UpSampling2D, Reshape, Activation, Conv2DTranspose
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import pickle as pkl
from tensorflow.python.layers import base

######## load the data set #########
# Change this data directory!!!!

os.getcwd()
filedirectory = "C:\\Users\\HeLi0001\\Desktop\\best_journeys\\"


def grab_random_gps_blocks(batchsize, blocksize=60):
    blocks = []
    blocks_collect = 0
    offset = blocksize // 2
    while blocks_collect < batchsize:
        file = random.choice(os.listdir(filedirectory))
        ds = pd.read_csv(filedirectory + file)
        for k in range(ds.shape[0] // blocksize):
            block = ds[k * blocksize: (k + 1) * blocksize].values
            start = np.copy(block[0, :])
            start[2] = 0
            start[3] = 0
            start[4] = 0
            start[6] = 0
            start[7] = 0
            start[8] = 0
            blocks.append(block - start)
            blocks_collect += 1
        for k in range((ds.shape[0] - offset) // blocksize):
            block = ds[k * blocksize + offset: (k + 1) * blocksize + offset].values
            start = np.copy(block[0, :])
            start[2] = 0
            start[3] = 0
            start[4] = 0
            start[6] = 0
            start[7] = 0
            start[8] = 0
            blocks.append(block - start)
            blocks_collect += 1
    blocks = random.sample(blocks, batchsize)
    return blocks


def data_sample_tensor(batchsize, blocksize):
    blocks = grab_random_gps_blocks(batchsize, blocksize)
    tnorm = blocksize - 1
    tblocks = np.zeros((batchsize, blocksize, 10))
    for foo, block in enumerate(blocks):
        tblocks[foo, :, 0] = block[:, 0] / tnorm
        tblocks[foo, :, 1] = block[:, 1] / tnorm / 27
        tblocks[foo, :, 2] = block[:, 2] / 27
        tblocks[foo, :, 3] = block[:, 3] / 27
        tblocks[foo, :, 4] = block[:, 4]
        tblocks[foo, :, 5] = block[:, 5] / tnorm / 27
        tblocks[foo, :, 6] = block[:, 6] / 27
        tblocks[foo, :, 7] = block[:, 7] / 27
        tblocks[foo, :, 8] = block[:, 8]
        tblocks[foo, :, 9] = block[:, 9] / 10
        # tblocks[foo, :, 0] = block[:, 0]
        # tblocks[foo, :, 1] = block[:, 1]
        # tblocks[foo, :, 2] = block[:, 2]
        # tblocks[foo, :, 3] = block[:, 3]
        # tblocks[foo, :, 4] = block[:, 4]
        # tblocks[foo, :, 5] = block[:, 5]
        # tblocks[foo, :, 6] = block[:, 6]
        # tblocks[foo, :, 7] = block[:, 7]
        # tblocks[foo, :, 8] = block[:, 8]
        # tblocks[foo, :, 9] = block[:, 9]
    return tblocks


sample_image = data_sample_tensor(30, 60)
print("input shape :", sample_image.shape)
sample_image = sample_image.reshape([-1,60,10,1])
print("input shape :", sample_image.shape)


def cre_generator():
    net = Sequential()
    dropout_prob = 0.1

    net.add(Dense(30*5*256, input_dim=182))  #5 + 3*59: numbers of parameters needed to calculate
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    net.add(Reshape((30, 5, 256)))
    #net.add(Dropout(dropout_prob))

    net.add(Conv2DTranspose(filters=1, kernel_size=5, strides=(2,2), padding='same'))
    net.add(Activation('tanh'))
    net.compile(loss='binary_crossentropy', optimizer='RMSprop')
    return net


g = cre_generator()
g.summary()


def cre_discriminator():
    net = Sequential()
    input_shape = (60,10,1)
    dropout_prob = 0.1

    net.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    #net.add(Dropout(dropout_prob))

    net.add(Conv2D(128, 5, strides=2, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    #net.add(Dropout(dropout_prob))

    net.add(Flatten())
    net.add(Dense(2, activation='tanh'))
    net.add(Activation('softmax'))
    net.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return net


d = cre_discriminator()
d.summary()


def create_gan(discriminator, generator):
    discriminator.trainable = False # only train the generator in combined model
    gan_input = Input(shape=(182,)) # keep same as the dim of generator input
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    return gan


gan = create_gan(d, g)
gan.summary()



blocksize = 60
epoch = 100
num_batches = 5

def training(epochs=1, batch_size=128):
    # Creating GAN
    generator = cre_generator()
    discriminator = cre_discriminator()
    gan = create_gan(discriminator, generator)
    D = []
    G = []
    for e in range(1, epochs + 1):
        avg_cost=0
        print("Epoch %d" % e)
        for _ in range(num_batches):
        # for _ in tqdm(range(batch_count)): # tqbm here is for the smart progress bar
            # generate  random noise as an input  to  initialize the  generator
            noise = np.random.rand(batch_size, 182) -1 # matrix nrow=batch_size, ncol=100

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            # ind = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
            # image_batch = X_train[ind]
            image_batch = data_sample_tensor(batch_size, blocksize).reshape([batch_size, blocksize, 10, 1])


            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images]) # join two data along row

            # Labels for generated and real data, real is 1 amd fake is 0
            y_dis = np.zeros([2 * batch_size, 2])
            # y_dis[:batch_size,:] = 1 # original is 0.9
            y_dis[0:batch_size, 1] = 1
            y_dis[batch_size:, 0] = 1
            # real_label = y_dis[:batch_size,1]
            # fake_label = y_dis[batch_size:,0]

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)
            # d_fake_loss = discriminator.train_on_batch(image_batch, real_label) # train_on_batch does one round train
            # d_real_loss = discriminator.train_on_batch(generated_images, fake_label)
            # d_loss = 0.5 * np.add(d_fake_loss, d_real_loss)
            D.append(d_loss)
            # Tricking the noised input of the Generator as real data
            # noise = np.random.rand(2 * batch_size, 182) -1 # original is 100
            noise = np.random.rand(batch_size, 182) - 1  # original is 100
            y_gen = np.zeros([batch_size, 2])
            y_gen[:, 1] = 1
            # y_gen = np.ones(2 * batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            g_loss = gan.train_on_batch(noise, y_gen)
            G.append(g_loss)
            print("Epoch %d Batch %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                  % (e, _, num_batches, d_loss[0], 100 * d_loss[1], g_loss))

    D = np.array(D) # change it into matrix
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(D[:,0]) # extract the first column
    plt.title('discriminant model loss')
    plt.ylabel('loss')
    plt.xlabel('batch')

    plt.subplot(2, 1, 2)
    # plt.plot(D[:,1]) # extract the second column
    # plt.title('discriminant model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('batch')
    plt.plot(G)
    plt.title("Generator model loss")
    plt.ylabel("loss")
    plt.xlabel("batch")

    plt.tight_layout()
    plt.show()
    # plt.savefig("evaluation GAN accuracy and loss %d" % _)
    # plt.close()


training(2, 128)