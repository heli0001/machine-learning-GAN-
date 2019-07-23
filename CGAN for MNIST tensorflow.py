import os
import tensorflow as tf
import numpy as np
import sys as os
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import pickle as pkl
from tensorflow.python.layers import base
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/")
sample_image = mnist.train.next_batch(1)[0]
print(sample_image.shape)
sample_image = sample_image.reshape([28, 28])
print("input shape :", sample_image.shape)


def generator(Z, reuse = False):
    with tf.variable_scope("GAN/Generator", reuse = reuse):
        h1 = tf.layers.dense(Z, 4*4*1024, use_bias=False)
        h1 = tf.reshape(h1, (-1,4,4,1024))
        bn0 = tf.layers.batch_normalization(h1, renorm_momentum=0.99)
        h1 = tf.layers.dropout(rate=0.1, inputs=bn0)
        # Deconvolution, 7*7*512
        conv1 = tf.layers.conv2d_transpose(h1, filters=512, kernel_size=4, strides=1, padding='valid')
        bn1 = tf.layers.batch_normalization(conv1, renorm_momentum=0.99)
        conv1 = tf.layers.dropout(rate=0.1, inputs=bn1)
        # Deconvolution, 14*14*256
        conv2 = tf.layers.conv2d_transpose(conv1, filters=256, kernel_size=5, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(conv2, renorm_momentum=0.99)
        conv2 = tf.layers.dropout(rate=0.1, inputs=bn2)
        # output layer, 28*28, where 1 is the out_channel_dim, since it is gray graph, if RCB colorful, then use 3
        out = tf.layers.conv2d_transpose(conv2, filters=1, kernel_size=5, strides=2, padding='same')
        out = tf.tanh(out)
    # def model_summary():
    #     model_vars = tf.trainable_variables()
    #     slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    #
    # model_summary()
    return out


# def model_summary():
#     model_vars = tf.trainable_variables()
#     slim.model_analyzer.analyze_vars(model_vars, print_info=True)
#
#
# model_summary()


def discriminator(X, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
    # Input layer is 28x28xn
    # Convolutional layer, 14x14x64
        X = tf.reshape(X,(-1,28,28,1))
        conv1 = tf.layers.conv2d(X, filters=64, kernel_size=5, strides=2, padding='same')
        conv1 = tf.layers.dropout(conv1, rate=0.1)
    # Strided convolutional layer, 7x7x128
        conv2 = tf.layers.conv2d(conv1, 128, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.layers.dropout(bn2, rate=0.1)
    # Strided convolutional layer, 4x4x256
        conv3 = tf.layers.conv2d(conv2, 256, 5, 2, 'same', use_bias=False)
        bn3 = tf.layers.batch_normalization(conv3)
        conv3 = tf.layers.dropout(bn3, rate=0.1)

    # fully connected
        flat = tf.reshape(conv3, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

    return out, logits


# Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,28,28,1])
Z = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,784])

G_sample = generator(Z)
r_rep, r_logits = discriminator(X)
f_rep, f_logits = discriminator(G_sample, reuse=True)
# y_rep, y_logits = discriminator(Y, reuse=True)

# Defining the loss functions of the network
smooth_factor =0.1
# disc_y_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))\
#               + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits, labels=tf.zeros_like(y_logits)))
disc_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))\
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

# Defining the optimizer
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(gen_loss, var_list= gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(disc_loss, var_list= disc_vars)


disc_rrate = tf.reduce_mean( r_logits)
disc_frate = tf.reduce_mean( f_logits)

# sess = tf.Session(config=config)
sess = tf.Session()
saver = tf.train.Saver() # save both variables
tf.global_variables_initializer().run(session=sess)

batch_size = 128

dlosses = []
glosses = []
rrates = []
frates = []
samples = []
pre_train = 3
epoch = 2
# for i in range(pre_train):
#     X_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
#     Z_batch = 2*np.random.rand(batch_size,784)-1
#     dloss_y = sess.run([disc_y_loss], feed_dict={X: X_batch, Y: Z_batch})
#     print(dloss_y)

for i in range(epoch):
    num_batches = mnist.train.num_examples//batch_size//150
    for i in range(num_batches):
        X_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        Z_batch = 2 * np.random.rand(batch_size, 784) - 1
        _, dloss, rrate, frate = sess.run([disc_step, disc_loss, disc_rrate, disc_frate], feed_dict={X: X_batch, Z: Z_batch})
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
        print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
        dlosses.append(dloss)
        glosses.append(gloss)
        rrates.append(rrate)
        frates.append(frate)
    Z_sample = 2 * np.random.rand(batch_size, 784) - 1
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
    plt.savefig('pics/gen_sample%d.png' % i)
    plt.close()


plt.plot(dlosses[:], label='D loss')
plt.plot(glosses[:], label='G loss')
plt.legend()
plt.show()

plt.plot(rrates[:], label='Mean real D score')
plt.plot(frates[:], label='Mean fake D score')
plt.legend()
plt.show()


# # example restore the variables
# tf.reset_default_graph()
# # Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "saved_model/gen.ckpt")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())
