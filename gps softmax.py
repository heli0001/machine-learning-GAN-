import pandas as pd
import numpy as np
import os
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gps_data_grab as gdb
import tensorflow.contrib.slim as slim
import pickle as pkl
from tensorflow.python.layers import base

######## load the data set #########


sample_image = gdb.data_sample_tensor(30, 60)
print("input shape :", sample_image.shape)
sample_image = sample_image.reshape([-1,60,10,1])
print("input shape :", sample_image.shape)


def generator(Z, reuse = False):
    with tf.variable_scope("GAN/Generator", reuse = reuse):
        h1 = tf.layers.dense(Z, 30*5*256, use_bias=False)
        h2 = tf.reshape(h1, (-1,30,5,256))
        bn0 = tf.layers.batch_normalization(h2, renorm_momentum=0.99)
        h3 = tf.layers.dropout(rate=0.1, inputs=bn0)
        h4 = tf.keras.layers.LeakyReLU(alpha=0.2)(h3)
        # output layer, 60*10, where 1 is the out_channel_dim, since it is gray graph, if RCB colorful, then use 3
        out = tf.layers.conv2d_transpose(h4, filters=1, kernel_size=5, strides=2, padding='same')
        out = tf.tanh(out)
    return out


def discriminator(X, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
    # Input layer is 60x10xn
    # Convolutional layer, 30x5x64
        X = tf.reshape(X,(-1,60,10,1))
        conv1 = tf.layers.conv2d(X, filters=64, kernel_size=5, strides=2, padding='same')
        bn1 = tf.layers.batch_normalization(conv1)
        a1 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn1)
        conv1up = tf.layers.dropout(a1, rate=0.1)
    # Strided convolutional layer, 15x3x128
        conv2 = tf.layers.conv2d(conv1up, 128, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2)
        a2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)
        conv2up = tf.layers.dropout(a2, rate=0.1)
    # fully connected
        flat = tf.reshape(conv2up, (-1, 15 * 3 * 128))
        logits = tf.layers.dense(flat, 2) # output of the final layer
 # It avoids overflows caused by taking the exp of large inputs and underflows caused by taking the log of small inputs.
        # gan_logits = tf.reduce_logsumexp(logits, 1)
        # out = tf.sigmoid(logits)
        out = tf.nn.softmax(logits)  # predicted output
        # classes = tf.argmax(input=logits, axis=1)
        # print(out.shape)
    return out, logits


batchsize = 128
blocksize = 60

# # Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,blocksize,10,1])

Z = tf.placeholder(tf.float32, [None,182])  # needed training parameter 5 + 59*3
# y = tf.placeholder(tf.float32, [None, 2]) # labels of the classes

G_sample = generator(Z)
print(G_sample.shape)
r_rep, r_logits = discriminator(X)
#print(r_logits.shape)
f_rep, f_logits = discriminator(G_sample, reuse=True)


# Defining the loss functions of the network
smooth_factor =0.1
disc_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
disc_loss = disc_loss_real + disc_loss_fake
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

# disc_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))
# disc_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=f_logits, labels=tf.zeros_like(f_logits)))
# disc_loss = disc_loss_real + disc_loss_fake
# gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=f_logits, labels=tf.ones_like(f_logits)))


# Defining the optimizer
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list= gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list= disc_vars)


disc_rrate = tf.reduce_mean( r_logits)
disc_frate = tf.reduce_mean( f_logits)

# sess = tf.Session(config=config)
sess = tf.Session()
saver = tf.train.Saver() # save both variables
tf.global_variables_initializer().run(session=sess)


dlosses = []
glosses = []
rrates = []
frates = []
samples = []
epoch = 5
num_batches = 50
for i in range(epoch):
    # num_batches = mnist.train.num_examples//batch_size//150
    for i in range(num_batches):
        # X_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        X_batch = gdb.data_sample_tensor(batchsize, blocksize).reshape([batchsize, blocksize, 10, 1])
        Z_batch = 2 * np.random.rand(batchsize, 182) - 1
        _, dloss, rrate, frate = sess.run([disc_step, disc_loss, disc_rrate, disc_frate], feed_dict={X: X_batch, Z: Z_batch})
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
        print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
        dlosses.append(dloss)
        glosses.append(gloss)
        rrates.append(rrate)
        frates.append(frate)
    Z_sample = 2 * np.random.rand(batchsize, 182) - 1
    gen_samples = sess.run(generator(Z, reuse=True), feed_dict={Z: Z_sample})
    print(gen_samples.shape)
    # plt.imshow(np.squeeze(gen_samples)[0], cmap='gray')
    # plt.savefig('pics/i%d.png' % i)
    # plt.show()
    samples.append(gen_samples) # saved as a list
    saver.save(sess, "saved_model/gen.ckpt")
for i in range(epoch):
    plt.imshow(np.squeeze(samples[i])[0], cmap='gray')
    plt.tight_layout()
    plt.savefig('pics/gps_gen_softmax_sample%d.png' % i)
    plt.close()


plt.plot(dlosses[:], label='D loss')
plt.plot(glosses[:], label='G loss')
plt.legend()
plt.show()

plt.plot(rrates[:], label='Mean real D score')
plt.plot(frates[:], label='Mean fake D score')
plt.legend()
plt.show()