import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    # x_train = x_train.astype(np.float32) / 255
    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)


(X_train, y_train, X_test, y_test) = load_data()
print(X_train.shape)


# use adam optimizer
def adam_optimizer():
    return Adam(lr=0.002, beta_1=0.5)


def create_generator():
    generator = Sequential()
    generator.add(Dense(256, input_dim=784))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784))
    generator.add(BatchNormalization())
    generator.add(Activation('tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator


g = create_generator()
g.summary() # summary of numbers of parameters needed to be trained


def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1))
    discriminator.add(BatchNormalization())
    discriminator.add(Activation('sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(), metrics=['accuracy'])
    return discriminator


d = create_discriminator()
d.summary()


def create_gan(discriminator, generator):
    discriminator.trainable = False # only train the generator in combined model
    gan_input = Input(shape=(784,)) # keep same as the dim of generator input
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


gan = create_gan(d, g)
gan.summary()


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' % epoch)


def training(epochs=1, batch_size=128):
    # Loading the data
    (X_train, y_train, X_test, y_test) = load_data()
    batch_count = int(X_train.shape[0] / batch_size)

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)
    D = []
    for e in range(1, epochs + 1):
        avg_cost=0
        print("Epoch %d" % e)
        for _ in range(batch_count):
        # for _ in tqdm(range(batch_count)): # tqbm here is for the smart progress bar
            # generate  random noise as an input  to  initialize the  generator
            noise = np.random.rand(batch_size, 784) -1 # matrix nrow=batch_size, ncol=100

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            ind = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
            image_batch = X_train[ind]

            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images]) # join two data along row

            # Labels for generated and real data, real is 1 amd fake is 0
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 1 # original is 0.9
            real_label = y_dis[:batch_size]
            fake_label = y_dis[batch_size:]

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            d_fake_loss = discriminator.train_on_batch(image_batch, real_label) # train_on_batch does one round train
            d_real_loss = discriminator.train_on_batch(generated_images, fake_label)
            d_loss = 0.5 * np.add(d_fake_loss, d_real_loss)
            D.append(d_loss)
            # Tricking the noised input of the Generator as real data
            noise = np.random.rand(2 * batch_size, 784) -1 # original is 100
            y_gen = np.ones(2 * batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            g_loss = gan.train_on_batch(noise, y_gen)
            print("Epoch %d Batch %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                  % (e, _, batch_count, d_loss[0], 100 * d_loss[1], g_loss))

    D = np.array(D) # change it into matrix
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(D[:,0]) # extract the first column
    plt.title('discriminant model loss')
    plt.ylabel('loss')
    plt.xlabel('batch')

    plt.subplot(2, 1, 2)
    plt.plot(D[:,1]) # extract the second column
    plt.title('discriminant model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('batch')

    plt.tight_layout()
    plt.show()
    # plt.savefig("evaluation GAN accuracy and loss %d" % _)
    # plt.close()


training(2, 256)
