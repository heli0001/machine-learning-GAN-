import os
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from tqdm import tqdm
import recon_gan_model as rgm

(X_all, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

# Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,28,28,1])
inputs = tf.placeholder(tf.float32, [None,28,28,1])
# Y = tf.placeholder(tf.float32, [None,28,28,1])

R_sample, noised_sample = rgm.reconstructor(inputs)
print("reconstructor shape", R_sample.shape)
r_rep, r_logits = rgm.discriminator(X)
f_rep, f_logits = rgm.discriminator(R_sample, reuse=True)

# Defining the loss functions of the network
smooth_factor =0.1
disc_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))\
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

epoch = 2
batch_size = 256
test_size = 2

# Running a new session
print("Starting 2nd session...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('training Recon GAN/final_model.ckpt.meta')
    # new_saver.restore(sess, "training Recon GAN/final_model.ckpt")
    new_saver.restore(sess, tf.train.latest_checkpoint('training Recon GAN/'))
    # inputs = tf.placeholder(tf.float32, [None,28,28,1])
    for i in range(epoch):
        num_batches = X_all.shape[0]//batch_size//35
        print("Epoch: %d" % i)
        for j in tqdm(range(num_batches)):
            ind = np.random.randint(low=0, high=X_all.shape[0], size=test_size)
            image_batch = X_all[ind]
            labels = y_train[ind]
            X_batch = image_batch.reshape([test_size, 28, 28,1])
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(np.squeeze(X_batch[0]), cmap='gray')
            # plt.title('random sample 1')
            # plt.subplot(1, 2, 2)
            # plt.imshow(np.squeeze(X_batch[1]), cmap='gray')
            # plt.title('random sample 2')
            # plt.tight_layout()
            # # plt.savefig('pics/recon_sample vs true_sample iteration %d.png' % i)
            # plt.show()
            recon_sample, _ = sess.run(rgm.reconstructor(inputs, reuse=True), feed_dict={inputs: X_batch})
            # recon_sample, _ = rgm.reconstructor(X_batch,reuse=True)
            out_prob, _ = sess.run(rgm.discriminator(inputs, reuse=True), feed_dict={inputs: recon_sample})
            # out_prob, _ = rgm.discriminator(recon_sample, reuse=True)
            print("Iterations: %d\t label1: %.4f\t label2: %.4f\t probability: %.4f\t probability: %.4f" %
                  (j, labels[0], labels[1], out_prob[0], out_prob[1]))