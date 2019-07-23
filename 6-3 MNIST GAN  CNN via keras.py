import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Input, Dropout, Flatten, BatchNormalization, UpSampling2D, Reshape, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import np_utils


#def load_data():
(X_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32) / 255
X_train = np.expand_dims(X_train, axis=3) # convert to 4D shape (60000, 28, 28, 1)
X_train = X_train.astype(np.float32) / 255


def build_generator(noise_shape=(100,)):
    inputx = Input(noise_shape)
    x = Dense(128 * 7 * 7)(inputx)
    x = Reshape((7, 7, 128))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # x = UpSampling2D()(x) # double the size to 14 * 14 * 128
    # padding = valid means no padding, =same means output same as input
    # number of filter=128, filter size =3,
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    # momentum for moving mean and variance
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(momentum=0.8)(x)
    # x = UpSampling2D()(x)
    # x = Conv2DTranspose(64, kernel_size=3, padding="same")(x)
    # x = BatchNormalization(momentum=0.8)(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(1, kernel_size=3, padding="same", strides=2, activation="tanh")(x)
    # x = BatchNormalization(momentum=0.8)(x)
    model = Model(inputx, x)
    print("-- Generator -- ")
    model.summary()
    return model


def build_discriminator(img_shape):
    inputx = Input(img_shape)
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(inputx)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    # x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = (LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
    out = Dense(1, activation="sigmoid")(x)
    # out = Dense(2, activation="softmax")(x)
    # out = BatchNormalization(momentum=0.8)(out)
    # out = Activation('sigmoid')(out)
    model = Model(inputx, out)
    print("-- Discriminator -- ")
    model.summary()
    return model


discriminator = build_discriminator(img_shape=(28, 28, 1))
generator = build_generator()
z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
real = discriminator(img)
combined = Model(z, real)

gen_optimizer = Adam(lr=0.0002, beta_1=0.5)
disc_optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=disc_optimizer, metrics=['accuracy'])
generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
combined.compile(loss='binary_crossentropy', optimizer=gen_optimizer)


def training(epochs=2, batch_size=256):
    num_examples = X_train.shape[0]
    num_batches = int(num_examples / float(batch_size))
    D = []
    for epoch in range(1, epochs + 1):
        for batch in range(num_batches):
    # noise images for the batch
            noise = np.random.normal(0, 1, (batch_size, 100))
            fake_images = generator.predict(noise)
            # fake_labels = np.zeros(batch_size)
            fake_labels = np.zeros((batch_size, 1)) # format as column vector
    # real images for batch
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]
            # real_labels = np.ones(batch_size)
            real_labels = np.ones((batch_size, 1))
    # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            D.append(d_loss)
            noise = np.random.normal(0, 1, (2*batch_size, 100))
    # Train the generator
            g_loss = combined.train_on_batch(noise, np.ones((2*batch_size, 1)))
    # Plot the progress
            print("Epoch %d Batch %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch,batch, num_batches, d_loss[0], 100 * d_loss[1], g_loss))
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


training(2, 256)