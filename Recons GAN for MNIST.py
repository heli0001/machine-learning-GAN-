import os
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

mnist = input_data.read_data_sets("MNIST_data/")
sample_image = mnist.train.next_batch(1)[0]
print(sample_image.shape)
sample_image = sample_image.reshape([28, 28])
print("input shape :", sample_image.shape)

std = 0.1
# seed = random.seed(1234)

#
#
# def gaussian_noise(inputs, std):
#     noise = tf.random_normal(shape = tf.shape(inputs), mean= 0, stddev= std, dtype= tf.float32,seed= seed)
#     return inputs + noise


def reconstructor(inputs, reuse=False):
    with tf.variable_scope('GAN/Reconstructor', reuse= reuse):
        with tf.variable_scope('GAN/encoder', reuse= reuse):
            # xc = tf.random_normal(shape=tf.shape(inputs), mean=0, stddev=std, dtype=tf.float32, seed=seed) + inputs
            xc = tf.random_normal(shape = tf.shape(inputs), mean= 0, stddev= std, dtype= tf.float32) + inputs
            xc = tf.reshape(xc, (-1, 28, 28, 1))
            # layer 1: 28,28,1 ---> 14,14,64
            conv1 = tf.layers.conv2d(xc, 64, 5, 2, 'same', use_bias=False)
            bn1 = tf.layers.batch_normalization(conv1)
            conv1_up = tf.nn.leaky_relu(bn1, alpha=0.1)
           # print("encoder layer1 shape", conv1.shape)
            # layer 2: 14,14,64 --->7,7,128
            conv2 = tf.layers.conv2d(conv1_up, 128, 5, 2, 'same', use_bias=False)
            bn2 = tf.layers.batch_normalization(conv2)
            conv2_up = tf.nn.leaky_relu(bn2, alpha=0.1)

            # layer 3: 7,7,128 --->4,4,256
            conv3 = tf.layers.conv2d(conv2_up, 256, 5, 2, 'same', use_bias=False)
            bn3 = tf.layers.batch_normalization(conv3)
            conv3_up = tf.nn.leaky_relu(bn3, alpha=0.1)

        with tf.variable_scope('GAN/decoder', reuse= reuse):
            # layer 1: 4,4,256 --->7,7,128
            dconv1 = tf.layers.conv2d_transpose(conv3_up, filters=128, kernel_size=4, strides=1, padding='valid')
            dbn1 = tf.layers.batch_normalization(dconv1, renorm_momentum=0.99)
            dconv1_up = tf.nn.relu(dbn1)

            # layer 2: 7,7,128 --->14,14,64
            dconv2 = tf.layers.conv2d_transpose(dconv1_up, filters=64, kernel_size=4, strides=2, padding='same')
            dbn2 = tf.layers.batch_normalization(dconv2, renorm_momentum=0.99)
            dconv2_up = tf.nn.relu(dbn2)

            # layer 3: 14,14,64 ---> 28,28,1
            out1 = tf.layers.conv2d_transpose(dconv2_up, filters=1, kernel_size=5, strides=2, padding='same')
            out = tf.tanh(out1)
    return out, xc


def discriminator(X, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
    # Input layer is 28x28xn
    # Convolutional layer, 14x14x64
        X = tf.reshape(X,(-1,28,28,1))
        conv1 = tf.layers.conv2d(X, filters=64, kernel_size=5, strides=2, padding='same')
        a1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
        conv1_up = tf.layers.dropout(a1, rate=0.1)
    # Strided convolutional layer, 7x7x128
        conv2 = tf.layers.conv2d(conv1_up, 128, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2)
        a2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)
        conv2_up = tf.layers.dropout(a2, rate=0.1)
    # Strided convolutional layer, 4x4x256
        conv3 = tf.layers.conv2d(conv2_up, 256, 5, 2, 'same', use_bias=False)
        bn3 = tf.layers.batch_normalization(conv3)
        a3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)
        conv3_up = tf.layers.dropout(a3, rate=0.1)

    # fully connected
        flat = tf.reshape(conv3_up, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
    return out, logits


# Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,28,28,1])
inputs = tf.placeholder(tf.float32, [None,28,28,1])
# Y = tf.placeholder(tf.float32, [None,28,28,1])

R_sample, noised_sample = reconstructor(inputs)
print("reconstructor shape", R_sample.shape)
r_rep, r_logits = discriminator(X)
f_rep, f_logits = discriminator(R_sample, reuse=True)
# y_rep, y_logits = discriminator(Y, reuse=True)

# Defining the loss functions of the network
smooth_factor =0.1
# disc_y_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))\
#               + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits, labels=tf.zeros_like(y_logits)))
disc_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))\
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

# Defining the optimizer
recon_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Reconstructor")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
recon_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(recon_loss, var_list= recon_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(disc_loss, var_list= disc_vars)

disc_rrate = tf.reduce_mean(r_logits)
disc_frate = tf.reduce_mean(f_logits)

# sess = tf.Session(config=config)
sess = tf.Session()
saver = tf.train.Saver() # save both variables
tf.global_variables_initializer().run(session=sess)

batch_size = 256

dlosses = []
relosses = []
rrates = []
frates = []
samples = []
input_samples = []
noise_input = []
# pre_train = 50
epoch = 3

# for i in tqdm(range(pre_train)):
#     X_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
#     Z_batch = X_batch
#     dloss_y = sess.run([disc_y_loss], feed_dict={X: X_batch, Y: Z_batch})
#     print(dloss_y)

for i in range(epoch):
    num_batches = mnist.train.num_examples//batch_size//10
    print("Epoch: %d" % i)
    for i in tqdm(range(num_batches)):
        X_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28,1])
        # print("input shape", X_batch.shape)
        # Z_batch = X_batch
        _, dloss, rrate, frate = sess.run([disc_step, disc_loss, disc_rrate, disc_frate], feed_dict={X: X_batch, inputs: X_batch})
        _, reloss = sess.run([recon_step, recon_loss], feed_dict={inputs: X_batch})
        print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, reloss))
        dlosses.append(dloss)
        relosses.append(reloss)
        rrates.append(rrate)
        frates.append(frate)
    Z_sample = mnist.train.next_batch(1)[0].reshape([1, 28, 28,1])
    recon_samples, noised_input = sess.run(reconstructor(inputs, reuse=True), feed_dict={inputs: Z_sample})
    print(recon_samples.shape)
    # plt.imshow(np.squeeze(gen_samples)[0], cmap='gray')
    # plt.savefig('pics/i%d.png' % i)
    # plt.show()
    input_samples.append(Z_sample)
    noise_input.append(noised_input)
    samples.append(recon_samples) # saved as a list
    saver.save(sess, "training Recon GAN/reGAN.ckpt")
for i in range(epoch):
    #if i % 10 == 0:
    plt.figure()
    plt.subplot(1, 3, 1)
    # plt.imshow(np.squeeze(samples[i])[0], cmap='gray')
    plt.imshow(np.squeeze(samples[i]), cmap='gray')
    plt.title('reconstructed sample')
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(input_samples[i]), cmap='gray')
    plt.title('real input sample')
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(noise_input[i]), cmap='gray')
    plt.title('noised input sample')
    plt.tight_layout()
    plt.savefig('pics/recon_sample vs true_sample iteration %d.png' % i)
    plt.close()


plt.plot(dlosses[:], label='D loss')
plt.plot(relosses[:], label='Re loss')
plt.legend()
plt.show()

plt.plot(rrates[:], label='Mean real D score')
plt.plot(frates[:], label='Mean fake D score')
plt.legend()
plt.show()